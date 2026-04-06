"""Utility functions for training scripts."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def get_checkpoint_dir(cfg: DictConfig, base: Path = Path('.')) -> Path:
    """Return the checkpoint directory based on dataset configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. Must contain ``cfg.dataset.bs_aggregation``
        (list of length 2 or 4) and ``cfg.dataset.full_cover`` (bool).
    base : Path, optional
        Root directory under which ``checkpoints/`` lives. Defaults to CWD.

    Returns
    -------
    Path
        Checkpoint directory path (not yet created).

    Raises
    ------
    ValueError
        If ``bs_aggregation`` length is not 2 or 4.
    """
    n_agents = len(cfg.dataset.bs_aggregation)
    full_cover = cfg.dataset.full_cover

    if n_agents == 2:
        subdir = '2_agents_full' if full_cover else '2_agents'
    elif n_agents == 4:
        subdir = '4_agents_full' if full_cover else '4_agents'
    else:
        raise ValueError(f'Unsupported bs_aggregation length {n_agents}. Expected 2 or 4.')

    return base / 'checkpoints' / subdir


def get_results_dir(cfg: DictConfig, base: Path = Path('.')) -> Path:
    """Return the per-run results directory based on dataset configuration.

    Mirrors the subdirectory logic of :func:`get_checkpoint_dir` but rooted
    under ``results/`` instead of ``checkpoints/``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    base : Path, optional
        Root directory. Defaults to CWD.

    Returns
    -------
    Path
        Results sub-directory path (not yet created).
    """
    n_agents = len(cfg.dataset.bs_aggregation)
    full_cover = cfg.dataset.full_cover

    if n_agents == 2:
        subdir = '2_agents_full' if full_cover else '2_agents'
    elif n_agents == 4:
        subdir = '4_agents_full' if full_cover else '4_agents'
    else:
        raise ValueError(f'Unsupported bs_aggregation length {n_agents}. Expected 2 or 4.')

    return base / 'results' / subdir


def save_orch_metrics(row: dict, orch_dir: Path) -> None:
    """Persist a single-orchestrator metrics row to ``{orch_dir}/eval_metrics.parquet``.

    Parameters
    ----------
    row : dict
        Flat metrics dictionary (one row of the evaluation DataFrame).
    orch_dir : Path
        Per-orchestrator results directory (created if absent).
    """
    import polars as pl

    orch_dir.mkdir(exist_ok=True, parents=True)
    pl.DataFrame([row]).write_parquet(orch_dir / 'eval_metrics.parquet')


def load_orch_metrics(orch_dir: Path) -> dict | None:
    """Load a previously saved per-orchestrator metrics row, if it exists.

    Parameters
    ----------
    orch_dir : Path
        Per-orchestrator results directory.

    Returns
    -------
    dict | None
        The metrics row as a plain Python dict, or ``None`` if no saved
        results are found in ``orch_dir``.
    """
    import polars as pl

    metrics_path = orch_dir / 'eval_metrics.parquet'
    if not metrics_path.exists():
        return None
    return pl.read_parquet(metrics_path).to_dicts()[0]


def save_checkpoint(
    orchestrator,
    cfg: DictConfig,
    trainer,
    logger,
    num_folds: int,
    base: Path = Path('.'),
) -> Path:
    """Save the orchestrator model checkpoint to the appropriate directory.

    The checkpoint directory is chosen based on ``cfg.dataset.bs_aggregation``
    length and ``cfg.dataset.full_cover``:

    * ``checkpoints/2_agents``      — 2 agents, full_cover=False
    * ``checkpoints/2_agents_full`` — 2 agents, full_cover=True
    * ``checkpoints/4_agents``      — 4 agents, full_cover=False
    * ``checkpoints/4_agents_full`` — 4 agents, full_cover=True

    Parameters
    ----------
    orchestrator :
        Trained orchestrator object to serialize with ``torch.save``.
    cfg : DictConfig
        Hydra configuration.
    trainer :
        Lightning Trainer (used to read ``callback_metrics``).
    logger :
        W&B logger instance (may be None).
    num_folds : int
        Number of training folds (appended to the checkpoint name).
    base : Path, optional
        Root directory under which ``checkpoints/`` lives. Defaults to CWD.

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    ckpt_dir = get_checkpoint_dir(cfg, base)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    project_name = cfg.logger.project
    run_name = str(logger.experiment.name) if logger is not None else 'run'
    lmb_schedule = cfg.orchestrator.get('lmb_schedule', None)
    if lmb_schedule is None:
        run_name = f'const_{run_name}'
    safe_run_name = run_name.replace('/', '_').replace(':', '-')

    train_loss = trainer.callback_metrics.get('train/total_loss', torch.tensor(0.0))
    if hasattr(train_loss, 'item'):
        train_loss = train_loss.item()

    max_epochs = cfg.trainer.max_epochs

    ckpt_name = (
        f'{project_name}_{safe_run_name}'
        f'_loss{train_loss:.4f}'
        f'_epochs{max_epochs}'
        f'_folds{num_folds}.pt'
    )
    ckpt_path = ckpt_dir / ckpt_name
    torch.save(orchestrator, ckpt_path)
    print(f'Model saved to {ckpt_path}')
    return ckpt_path


def find_best_checkpoint(
    ckpt_dir: Path,
    project_name: str,
    orchestrator_name: str,
) -> Path | None:
    """Find the checkpoint with the lowest training loss for a given orchestrator.

    Searches ``ckpt_dir`` for ``.pt`` files whose name starts with
    ``{project_name}_{orchestrator_name}_`` and picks the one with the
    smallest loss value encoded as ``_loss{value}_`` in the filename.

    Parameters
    ----------
    ckpt_dir : Path
        Directory containing checkpoint files.
    project_name : str
        W&B project name prefix used in the checkpoint filename.
    orchestrator_name : str
        Hydra orchestrator config name (e.g. ``'cover_sheaf'``).

    Returns
    -------
    Path | None
        Path to the best checkpoint, or ``None`` if no matching file is found.
    """
    loss_pattern = re.compile(r'_loss(\d+\.\d+)_')
    # Accept both '{project}_{orch}_...' and '{project}_const_{orch}_...'
    name_pattern = re.compile(
        rf'^{re.escape(project_name)}_(?:const_)?{re.escape(orchestrator_name)}_'
    )
    candidates: list[tuple[float, Path]] = []
    for p in ckpt_dir.glob('*.pt'):
        if not name_pattern.match(p.stem):
            continue
        m = loss_pattern.search(p.name)
        if m:
            candidates.append((float(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class _FakeTrainer:
    """Minimal trainer stub that satisfies ``orchestrator.trainer.datamodule``."""

    def __init__(self, datamodule: Any) -> None:
        self.datamodule = datamodule
        self.max_epochs = 1


def compute_loss_metrics(orchestrator, datamodule) -> dict[str, float]:
    """Run the model on the training dataloader and collect loss metrics.

    Uses a minimal Lightning Trainer (no logger, no checkpointing) to execute
    ``test_step`` on the training split.  All metrics logged under the
    ``test/`` prefix are returned with that prefix stripped, so callers get
    plain names like ``total_loss``, ``total_private_loss``,
    ``total_alignment_loss``, ``triplet_loss_agent_0``, etc.

    Parameters
    ----------
    orchestrator :
        Loaded orchestrator object (``BaseOrchestrator`` subclass).
    datamodule :
        Fully set-up ``DICHASUSDataModule`` (``setup('fit')`` already called).

    Returns
    -------
    dict[str, float]
        Metric name → scalar value, with the ``test/`` prefix removed.
    """
    from lightning import Trainer

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=0,
    )
    trainer.test(orchestrator, dataloaders=datamodule.train_dataloader())

    return {
        k.removeprefix('test/'): v.item() if hasattr(v, 'item') else float(v)
        for k, v in trainer.callback_metrics.items()
        if k.startswith('test/')
    }


@torch.no_grad()
def compute_eval_metrics(
    orchestrator,
    datamodule,
    K_max: int = 10,
    K_min: int = 2,
    step: int = 1,
) -> dict[str, Any]:
    """Compute evaluation metrics without W&B logging.

    Replicates the computation done by ``BaseOrchestrator.eval_all`` but
    does not attempt to log anything to W&B. Works on any loaded checkpoint
    by attaching a minimal fake trainer stub.

    Metrics computed per agent:
    - **KS**: Kruskal stress (scalar per agent).
    - **CT[K]**: Continuity at neighbour count K (scalar per agent).
    - **TW[K]**: Trustworthiness at neighbour count K (scalar per agent).

    Metrics computed globally:
    - **FOSCTTM**: alignment quality across edges (``None`` for orchestrators
      that have not yet implemented this metric).

    Parameters
    ----------
    orchestrator :
        Loaded orchestrator object (``BaseOrchestrator`` subclass).
    datamodule :
        Fully set-up ``DICHASUSDataModule`` (``setup('fit')`` already called).
    K_max : int
        Maximum number of nearest neighbours.
    K_min : int
        Minimum number of nearest neighbours.
    step : int
        Step between K values.

    Returns
    -------
    dict
        ``{'KS': list[float], 'CT': {K: list[float]}, 'TW': {K: list[float]},
        'FOSCTTM': float | None}``
    """
    orchestrator._trainer = _FakeTrainer(datamodule)
    orchestrator.eval()

    res: dict[str, Any] = {
        'KS': [],
        'CT': defaultdict(list),
        'TW': defaultdict(list),
    }

    for agent_idx in orchestrator.agents:
        embs, pos, embs_KDTree, pos_KDTree = orchestrator.build_trajectory(
            agent_idx=int(agent_idx),
            split='train',
        )
        res['KS'].append(orchestrator.compute_kruskal_stress(embs=embs, pos=pos).item())
        for K in range(K_min, K_max + 1, step):
            res['CT'][K].append(
                orchestrator.compute_continuity(
                    embs=embs, pos=pos, pos_KDTree=pos_KDTree, K=K
                ).item()
            )
            res['TW'][K].append(
                orchestrator.compute_trustworthiness(
                    embs=embs, pos=pos, embs_KDTree=embs_KDTree, K=K
                ).item()
            )

    foscttm = orchestrator._compute_FOSCTTM()
    if foscttm is not None and hasattr(foscttm, 'item'):
        res['FOSCTTM'] = foscttm.item()
    else:
        res['FOSCTTM'] = None

    return res


@torch.no_grad()
def save_latent_representations(
    orchestrator,
    datamodule,
    orch_dir: Path,
    batch_size: int = 256,
) -> None:
    """Save latent representations for all local and shared datasets.

    For each agent, the full local-dataset embeddings and positions are saved.
    For each edge (i, j) in the shared dataset, the embeddings produced by
    agent i (from BS-i CSI) and agent j (from BS-j CSI) are saved.

    Output files are written directly into ``orch_dir``:

    * ``local_agent_{idx}.pt``  →  ``{'embs': Tensor(N, d), 'pos': Tensor(N, 2)}``
    * ``shared_{i}_{j}.pt``     →  ``{'embs_i': Tensor(M, d), 'embs_j': Tensor(M, d)}``

    Parameters
    ----------
    orchestrator :
        Loaded orchestrator (``BaseOrchestrator`` subclass), already in eval mode.
    datamodule :
        Fully set-up ``DICHASUSDataModule`` (``setup('fit')`` already called).
    orch_dir : Path
        Per-orchestrator results directory (e.g.
        ``Path('results/2_agents/optimal_transport')``). Created if absent.
    batch_size : int
        Batch size used when running the shared-dataset forward pass.
    """
    orch_dir.mkdir(exist_ok=True, parents=True)
    orch_name = orch_dir.name

    # Attach a minimal trainer stub so build_trajectory can reach the datamodule
    orchestrator._trainer = _FakeTrainer(datamodule)
    orchestrator.eval()

    # ------------------------------------------------------------------
    # Local representations — one file per agent
    # ------------------------------------------------------------------
    for agent_idx_str in orchestrator.agents:
        agent_idx = int(agent_idx_str)
        embs, pos, _, _ = orchestrator.build_trajectory(agent_idx=agent_idx, split='train')
        out_path = orch_dir / f'local_agent_{agent_idx}.pt'
        torch.save({'embs': embs.cpu(), 'pos': pos.cpu()}, out_path)
        print(f'  [{orch_name}] local agent {agent_idx} → {out_path}  shape={tuple(embs.shape)}')

    # ------------------------------------------------------------------
    # Shared representations — one file per edge (i, j)
    # ------------------------------------------------------------------
    for (i, j), shared_ds in datamodule.train_shared_dataset.items():
        agent_i = orchestrator.agents[str(i)]
        agent_j = orchestrator.agents[str(j)]

        loader = DataLoader(shared_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        embs_i_list: list[torch.Tensor] = []
        embs_j_list: list[torch.Tensor] = []

        for H_1, H_2, _ in loader:
            # agent.forward with triplet_mode=False encodes a raw tensor directly
            embs_i_list.append(agent_i(H_1, triplet_mode=False).cpu())
            embs_j_list.append(agent_j(H_2, triplet_mode=False).cpu())

        embs_i = torch.cat(embs_i_list, dim=0)
        embs_j = torch.cat(embs_j_list, dim=0)

        out_path = orch_dir / f'shared_{i}_{j}.pt'
        torch.save({'embs_i': embs_i, 'embs_j': embs_j}, out_path)
        print(f'  [{orch_name}] shared ({i},{j}) → {out_path}  shape={tuple(embs_i.shape)}')
