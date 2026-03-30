"""Single-agent training script for DICHASUS dataset.

Runs the full pipeline for one base station without an orchestrator or
CombinedLoader.  The Agent is wrapped directly in SingleAgentModule, which
acts as the LightningModule.  Data is served by DICHASUSSingleAgentDataModule,
which returns plain DataLoaders for base station 0's local dataset.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.datamodules import DICHASUSSingleAgentDataModule
from src.utils import remove_non_empty_dir


@hydra.main(
    config_path='../config/hydra/',
    config_name='single_agent_train',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    CURRENT = Path('.')
    RESULTS_PATH = CURRENT / 'results/'
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
    if cfg.callbacks is None:
        callbacks = []
    else:
        callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    # ===================================================
    #             Define the DataModule (first fold)
    # ===================================================
    datamodule = DICHASUSSingleAgentDataModule(
        cfg.dataset, anchor_seed=anchor_seed, triplet_seed=triplet_seeds[0]
    )
    datamodule.prepare_data()
    datamodule.setup('fit')

    # ===================================================
    #         Define Agent + SingleAgentModule
    # ===================================================
    agent = instantiate(cfg.model, in_dim=datamodule.feature_dim)

    module = instantiate(
        cfg.orchestrator,
        agent=agent,
        _convert_='all',
    )

    # -------------------------
    # Multi-fold Training
    # -------------------------
    for fold_idx in range(num_folds):
        print(f'\n{"=" * 50}')
        print(f'Starting Fold {fold_idx + 1}/{num_folds}')
        print(f'  triplet_seed: {triplet_seeds[fold_idx]}')
        print(f'  learning_rate: {learning_rates[fold_idx]}')
        print(f'{"=" * 50}\n')

        # Set learning rate for this fold
        module.set_lr(learning_rates[fold_idx])

        # For fold > 0, regenerate datamodule with new triplet seed
        if fold_idx > 0:
            datamodule = DICHASUSSingleAgentDataModule(
                cfg.dataset, anchor_seed=anchor_seed, triplet_seed=triplet_seeds[fold_idx]
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
        trainer.fit(module, datamodule=datamodule)

    # ===================================================
    #                     Evaluate
    # ===================================================
    module.eval_all(K_max=10)

    # Cleaning the working space
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)


if __name__ == '__main__':
    main()
