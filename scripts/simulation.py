""""""

# Add root to the path
import math
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from collections import defaultdict

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

# import omegaconf
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from scripts.util import (
    compute_eval_metrics,
    compute_loss_metrics,
    get_results_dir,
    save_checkpoint,
    save_latent_representations,
    save_orch_metrics,
)
from src.datamodules.dichasus import DICHASUSDataModule
from src.utils import remove_non_empty_dir


@hydra.main(
    config_path='../config/hydra/',
    config_name='train',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main simulation loop."""

    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    # Define some usefull paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'

    # Create directories
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # ===================================================
    #             Fold Configuration
    # ===================================================
    num_folds = cfg.get('num_folds', 1)
    anchor_seed = cfg.get('anchor_seed', cfg.seed)
    triplet_seeds = cfg.get('triplet_seeds', list(range(num_folds)))
    learning_rates = cfg.get('learning_rates', [cfg.get('lr', 1e-3)] * num_folds)
    separate_fold_runs = cfg.get('separate_fold_runs', False)

    if len(learning_rates) != num_folds:
        raise ValueError(
            f'learning_rates length ({len(learning_rates)}) must match num_folds ({num_folds})'
        )
    if len(triplet_seeds) != num_folds:
        raise ValueError(
            f'triplet_seeds length ({len(triplet_seeds)}) must match num_folds ({num_folds})'
        )

    # ===================================================
    #                  Wandb Logger
    # ===================================================
    logger = instantiate(cfg.logger)

    # ===================================================
    #             Define the Trainer
    # ===================================================
    # Instantiate callbacks
    if cfg.callbacks is None:
        callbacks = []
    else:
        callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    # ===================================================
    #             Define the DataModule (first fold)
    # ===================================================
    datamodule = DICHASUSDataModule(
        cfg.dataset,
        anchor_seed=anchor_seed,
        triplet_seed=triplet_seeds[0],
    )
    datamodule.prepare_data()
    datamodule.setup('fit')

    # datamodule.plot_trajectories(n=4, stage="train", plot_name="prova")

    agents = {}

    for i in range(datamodule.n_agents):
        agents[i] = instantiate(cfg.model, in_dim=datamodule.feature_dim)

    # ===================================================
    #                Define the Orchestrator
    # ===================================================
    neighbors = defaultdict(set)

    for u, v in cfg.dataset.edge_set:
        neighbors[u].add(v)
        neighbors[v].add(u)  # remove this line if graph is directed

    neighbors = dict(neighbors)

    # Instantiate orchestrator
    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        _convert_='all',
    )

    # ===================================================
    #        External Lambda Schedule (across folds)
    # ===================================================
    # If the orchestrator has a lambda schedule (lmb_schedule is not None),
    # build an external cosine schedule that partitions [lmb_min, lmb_max]
    # into num_folds intervals.  Fold i then runs its internal scheduler
    # inside [external_lambdas[i], external_lambdas[i+1]].
    lmb_schedule = cfg.orchestrator.get('lmb_schedule', None)
    if lmb_schedule is not None and num_folds > 1:
        lmb_min_global = float(cfg.orchestrator.get('lmb_min', 0.0))
        lmb_max_global = float(cfg.orchestrator.get('lmb_max', 1.0))
        external_lambdas = [
            lmb_min_global
            + (lmb_max_global - lmb_min_global) * (1 - math.cos(math.pi * i / num_folds)) / 2
            for i in range(num_folds + 1)
        ]
    else:
        external_lambdas = None

    # -------------------------
    # Multi-fold Training
    # -------------------------
    for fold_idx in range(num_folds):
        print(f'\n{"=" * 50}')
        print(f'Starting Fold {fold_idx + 1}/{num_folds}')
        print(f'  triplet_seed: {triplet_seeds[fold_idx]}')
        print(f'  learning_rate: {learning_rates[fold_idx]}')
        if external_lambdas is not None:
            lmb = external_lambdas
            print(f'  lmb_range: [{lmb[fold_idx]:.4f}, {lmb[fold_idx + 1]:.4f}]')
        print(f'{"=" * 50}\n')

        # Set learning rate for this fold
        orchestrator.set_lr(learning_rates[fold_idx])

        # Apply external lambda range for this fold
        if external_lambdas is not None:
            orchestrator.set_lmb_range(
                lmb_min=external_lambdas[fold_idx],
                lmb_max=external_lambdas[fold_idx + 1],
            )

        # For fold > 0, regenerate datamodule with new triplet seed
        if fold_idx > 0:
            datamodule = DICHASUSDataModule(
                cfg.dataset,
                anchor_seed=anchor_seed,
                triplet_seed=triplet_seeds[0],
            )
            datamodule.prepare_data()
            datamodule.setup('fit')

        # Create logger for this fold
        if separate_fold_runs and logger is not None:
            from lightning.pytorch.loggers import WandbLogger

            fold_logger = WandbLogger(
                project=cfg.logger.project,
                name=f'{cfg.logger.name}_fold{fold_idx + 1}',
                log_model=False,
            )
            fold_logger.experiment.config.update(
                OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
            )
        else:
            fold_logger = logger

        # Create a new Trainer for each fold to reset state
        trainer = Trainer(
            **cfg.trainer,
            callbacks=callbacks,
            logger=fold_logger,
        )

        # Train on this fold
        trainer.fit(orchestrator, datamodule=datamodule)

    # ===================================================
    #                  Save Model
    # ===================================================
    if cfg.get('save_model', False):
        save_checkpoint(orchestrator, cfg, trainer, logger, num_folds, base=CURRENT)

    # ===================================================
    #              Evaluate and Save Metrics
    # ===================================================
    K_max: int = cfg.get('eval_K_max', 40)
    K_min: int = cfg.get('eval_K_min', 2)
    step: int = cfg.get('eval_K_step', 4)

    orch_name: str = HydraConfig.get().runtime.choices.get('orchestrator', 'unknown')
    results_subdir = get_results_dir(cfg, CURRENT)
    results_subdir.mkdir(exist_ok=True, parents=True)
    orch_dir = results_subdir / orch_name

    print(f'\n[{orch_name}] Computing losses on training data…')
    loss_metrics = compute_loss_metrics(orchestrator, datamodule)

    print(f'[{orch_name}] Computing topology metrics…')
    metrics = compute_eval_metrics(
        orchestrator,
        datamodule,
        K_max=K_max,
        K_min=K_min,
        step=step,
    )

    row: dict = {'orchestrator': orch_name}
    row.update(loss_metrics)

    for i, ks_val in enumerate(metrics['KS']):
        row[f'KS_agent_{i}'] = float(ks_val)
    row['KS_mean'] = float(sum(metrics['KS']) / len(metrics['KS'])) if metrics['KS'] else None

    for K in range(K_min, K_max + 1, step):
        ct_vals = metrics['CT'][K]
        tw_vals = metrics['TW'][K]
        row[f'CT_K{K}'] = float(sum(ct_vals) / len(ct_vals)) if ct_vals else None
        row[f'TW_K{K}'] = float(sum(tw_vals) / len(tw_vals)) if tw_vals else None

    row['FOSCTTM'] = float(metrics['FOSCTTM']) if metrics['FOSCTTM'] is not None else None

    total_loss = loss_metrics.get('total_loss', float('nan'))
    print(
        f'[{orch_name}] total_loss={total_loss:.4f}  '
        f'KS={row["KS_mean"]:.4f}  FOSCTTM={row["FOSCTTM"]}'
    )

    save_orch_metrics(row, orch_dir)

    print(f'[{orch_name}] Saving latent representations…')
    try:
        save_latent_representations(orchestrator, datamodule, orch_dir)
    except Exception as exc:
        print(f'[{orch_name}] Latent representation saving failed: {exc}')

    df = pl.DataFrame([row])
    agg_path = results_subdir / 'eval_metrics.parquet'
    df.write_parquet(agg_path)
    print(f'\nSaved aggregated metrics → {agg_path}')
    print(df)

    # Cleaning the working space
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)

    return None


if __name__ == '__main__':
    main()
