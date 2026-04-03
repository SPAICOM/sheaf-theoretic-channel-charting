"""Evaluation script: load saved checkpoints and compute metrics for all orchestrators.

Uses the same Hydra config as ``simulation.py``.  For each orchestrator type
the script:

1. Resolves the per-run results directory from the dataset config (same subdir
   logic used for checkpoints: 2_agents, 4_agents, etc.).
2. If ``results/{subdir}/{orch_name}/eval_metrics.parquet`` already exists,
   the orchestrator is **skipped** and its cached metrics + latent
   representations are reused.
3. Otherwise, finds the checkpoint with the lowest encoded training loss,
   loads the full orchestrator, evaluates it on the training dataset, and
   saves both metrics and latent representations to the per-orchestrator dir.
4. Collects KS, CT@K, TW@K, and FOSCTTM metrics for every agent.

Per-orchestrator results are written to:
    ``results/{subdir}/{orch_name}/eval_metrics.parquet``
    ``results/{subdir}/{orch_name}/local_agent_{i}.pt``
    ``results/{subdir}/{orch_name}/shared_{i}_{j}.pt``

An aggregated summary across all orchestrators is also written to:
    ``results/{subdir}/eval_metrics.parquet``
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
    get_results_dir,
    load_orch_metrics,
    save_latent_representations,
    save_orch_metrics,
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
    #   Directories
    # ===================================================
    ckpt_dir = get_checkpoint_dir(cfg, CURRENT)
    results_subdir = get_results_dir(cfg, CURRENT)  # e.g. results/2_agents
    results_subdir.mkdir(exist_ok=True, parents=True)

    project_name = cfg.logger.project

    K_max: int = cfg.get('eval_K_max', 40)
    K_min: int = cfg.get('eval_K_min', 2)
    step: int = cfg.get('eval_K_step', 4)

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
    #   Evaluate each orchestrator
    # ===================================================
    records: list[dict] = []

    for orch_name in ORCHESTRATOR_NAMES:
        orch_dir = results_subdir / orch_name

        # --------------------------------------------------
        # Cache hit: reuse existing per-orchestrator results
        # --------------------------------------------------
        cached_row = load_orch_metrics(orch_dir)
        if cached_row is not None:
            print(f'\n[{orch_name}] Found existing results in {orch_dir} — skipping evaluation.')
            records.append(cached_row)
            continue

        # --------------------------------------------------
        # No cache: locate checkpoint
        # --------------------------------------------------
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
        row['KS_mean'] = float(sum(metrics['KS']) / len(metrics['KS'])) if metrics['KS'] else None

        # Per-K continuity and trustworthiness (mean across agents)
        for K in range(K_min, K_max + 1, step):
            ct_vals = metrics['CT'][K]
            tw_vals = metrics['TW'][K]
            row[f'CT_K{K}'] = float(sum(ct_vals) / len(ct_vals)) if ct_vals else None
            row[f'TW_K{K}'] = float(sum(tw_vals) / len(tw_vals)) if tw_vals else None

        # Global alignment metric
        row['FOSCTTM'] = float(metrics['FOSCTTM']) if metrics['FOSCTTM'] is not None else None

        total_loss = loss_metrics.get('total_loss', float('nan'))
        print(
            f'[{orch_name}] total_loss={total_loss:.4f}  '
            f'KS={row["KS_mean"]:.4f}  FOSCTTM={row["FOSCTTM"]}'
        )

        # --------------------------------------------------
        # Persist per-orchestrator metrics and latent reps
        # --------------------------------------------------
        save_orch_metrics(row, orch_dir)

        print(f'[{orch_name}] Saving latent representations…')
        try:
            save_latent_representations(orchestrator, datamodule, orch_dir)
        except Exception as exc:
            print(f'[{orch_name}] Latent representation saving failed: {exc}')

        records.append(row)

    if not records:
        print('No orchestrators evaluated. Nothing to save.')
        return

    # ===================================================
    #   Aggregated summary across all orchestrators
    # ===================================================
    df = pl.DataFrame(records)
    agg_path = results_subdir / 'eval_metrics.parquet'
    df.write_parquet(agg_path)

    print(f'\nSaved aggregated metrics → {agg_path}')
    print(df)


if __name__ == '__main__':
    main()
