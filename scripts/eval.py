"""Evaluation script: load saved checkpoints and compute metrics for all orchestrators.

Uses the same Hydra config as ``simulation.py``.  For each orchestrator type
the script:

1. Resolves the checkpoint directory from the dataset config (same logic used
   when saving).
2. Finds the checkpoint with the lowest encoded training loss.
3. Loads the full orchestrator object and evaluates it on the *training*
   dataset (the only split available for DICHASUS cf0x when ``test_split=0``).
4. Collects KS, CT@K, TW@K, and FOSCTTM metrics for every agent.

All results are written to ``results/eval_metrics.parquet`` as a wide-format
Polars DataFrame with one row per orchestrator and one column per metric.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import polars as pl
import torch
from lightning import seed_everything
from omegaconf import DictConfig

from scripts.util import (
    compute_eval_metrics,
    compute_loss_metrics,
    find_best_checkpoint,
    get_checkpoint_dir,
)
from src.datamodules.dichasus import DICHASUSDataModule

# All orchestrator config names that may have a saved checkpoint
ORCHESTRATOR_NAMES = [
    # 'bundle',
    # 'cover_sheaf',
    # 'diag_sheaf',
    # 'federated',
    # 'flat_bundle',
    # 'neural_diag_sheaf',
    'optimal_transport',
    # 'personalized_federated',
    # 'single_agent',
    # 'vanilla',
]


@hydra.main(
    config_path='../config/hydra/',
    config_name='train',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    CURRENT = Path('.')

    # ===================================================
    #   DataModule — initialised once, shared across all
    #   orchestrators evaluated in this run
    # ===================================================
    datamodule = DICHASUSDataModule(
        cfg.dataset,
        anchor_seed=cfg.get('anchor_seed', cfg.seed),
        triplet_seed=cfg.get('triplet_seeds', [cfg.seed])[0],
    )
    datamodule.prepare_data()
    datamodule.setup('fit')

    # ===================================================
    #   Checkpoint directory (mirrors save_checkpoint logic)
    # ===================================================
    ckpt_dir = get_checkpoint_dir(cfg, CURRENT)
    project_name = cfg.logger.project

    K_max: int = cfg.get('eval_K_max', 10)
    K_min: int = cfg.get('eval_K_min', 2)
    step: int = cfg.get('eval_K_step', 1)

    # ===================================================
    #   Evaluate each orchestrator
    # ===================================================
    records: list[dict] = []

    for orch_name in ORCHESTRATOR_NAMES:
        ckpt_path = find_best_checkpoint(ckpt_dir, project_name, orch_name)
        if ckpt_path is None:
            print(f'[{orch_name}] No checkpoint found in {ckpt_dir} — skipping.')
            continue

        print(f'\n[{orch_name}] Loading {ckpt_path.name}')
        orchestrator = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        orchestrator.eval()

        print(f'[{orch_name}] Computing losses on training data…')
        try:
            loss_metrics = compute_loss_metrics(orchestrator, datamodule)
        except Exception as exc:
            print(f'[{orch_name}] Loss computation failed: {exc} — skipping.')
            continue

        print(f'[{orch_name}] Computing topology metrics…')
        try:
            metrics = compute_eval_metrics(
                orchestrator,
                datamodule,
                K_max=K_max,
                K_min=K_min,
                step=step,
            )
        except Exception as exc:
            print(f'[{orch_name}] Metric computation failed: {exc} — skipping.')
            continue

        # --------------------------------------------------
        # Flatten metrics into a single dict (one DataFrame row)
        # --------------------------------------------------
        row: dict = {'orchestrator': orch_name}

        # Loss metrics from training-data forward pass
        # Includes: total_loss, total_private_loss, total_alignment_loss,
        #           triplet_loss_agent_i, rec_loss_agent_i (when decoder is used)
        row.update(loss_metrics)

        # Per-agent Kruskal stress
        for i, ks_val in enumerate(metrics['KS']):
            row[f'KS_agent_{i}'] = float(ks_val)
        row['KS_mean'] = (
            float(sum(metrics['KS']) / len(metrics['KS'])) if metrics['KS'] else None
        )

        # Per-K continuity and trustworthiness (mean across agents)
        for K in range(K_min, K_max + 1, step):
            ct_vals = metrics['CT'][K]
            tw_vals = metrics['TW'][K]
            row[f'CT_K{K}'] = float(sum(ct_vals) / len(ct_vals)) if ct_vals else None
            row[f'TW_K{K}'] = float(sum(tw_vals) / len(tw_vals)) if tw_vals else None

        # Global alignment metric
        row['FOSCTTM'] = float(metrics['FOSCTTM']) if metrics['FOSCTTM'] is not None else None

        records.append(row)
        total_loss = loss_metrics.get('total_loss', float('nan'))
        print(
            f'[{orch_name}] total_loss={total_loss:.4f}  '
            f'KS={row["KS_mean"]:.4f}  FOSCTTM={row["FOSCTTM"]}'
        )

    if not records:
        print('No orchestrators evaluated. Nothing to save.')
        return

    # ===================================================
    #   Build Polars DataFrame and persist
    # ===================================================
    df = pl.DataFrame(records)

    results_dir = CURRENT / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)
    out_path = results_dir / 'eval_metrics.parquet'
    df.write_parquet(out_path)

    print(f'\nSaved evaluation metrics → {out_path}')
    print(df)


if __name__ == '__main__':
    main()
