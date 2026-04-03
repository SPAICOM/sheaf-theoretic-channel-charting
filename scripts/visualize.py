"""Visualization script: load checkpoints and plot latent representations.

Uses the same Hydra config as ``eval.py``.  For each orchestrator:

1. Finds the best checkpoint and loads the full orchestrator object.
2. Reads the pre-saved latent representations from the per-orchestrator
   results directory (``local_agent_{i}.pt`` and ``shared_{i}_{j}.pt``).
3. Produces side-by-side **before / after alignment** figures:

   - **Group 1** (``flat_bundle``, ``federated``, ``personalized_federated``):
     Left panel = raw per-agent local latents (unaligned coordinate frames).
     Right panel = aligned latents:
       * ``flat_bundle``  → ``embs @ R.T`` where R is ``local_reference_frames``
       * ``federated`` / ``personalized_federated`` → same as raw (FedAvg already
         drives agents toward a shared parameter space; no explicit post-hoc
         transform exists)

   - **Group 2** (all other orchestrators):
     One row per edge (i, j); left col = raw shared embeddings, right col =
     both sides after applying the learned alignment map:
       * ``bundle``            – i raw, j → i's space via ``O_ij.T``
                                  (training objective: ``‖emb_i − emb_j @ O_ij.T‖``)
       * ``diag_sheaf``        – i raw, j → i's space via ``D_ij``
                                  (D is diagonal so D.T = D;
                                   objective: ``‖emb_i − emb_j @ D_ij‖``)
       * ``cover_sheaf``       – both raw (identity alignment, no map to apply)
       * ``neural_diag_sheaf`` – ``D_i(emb_i)`` and ``D_j(emb_j)``
                                  (objective: ``‖D_i(emb_i) − D_j(emb_j)‖``)
       * ``optimal_transport`` – ``T_i(emb_i)`` and ``T_j(emb_j)``
                                  (objective: ``‖T_i(emb_i) − T_j(emb_j)‖``)
       * ``vanilla``, ``single_agent`` – both raw (no map)

     Agent colors are consistent across all rows so the same agent always
     appears in the same color.

Output files:
    ``results/{subdir}/{orch_name}/latent_combined.png``   (Group 1)
    ``results/{subdir}/{orch_name}/latent_edges.png``      (Group 2)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import seed_everything
from omegaconf import DictConfig

from scripts.util import find_best_checkpoint, get_checkpoint_dir, get_results_dir

matplotlib.use('Agg')

STYLE_PATH = Path(__file__).resolve().parents[1] / 'config/plotting/plt.mplstyle'
plt.style.use(str(STYLE_PATH))

# Orchestrators that share (or can be synced into) a single global latent space
GROUP1_ORCHS = {'flat_bundle', 'federated', 'personalized_federated'}

# All orchestrators that may have saved checkpoints
ORCHESTRATOR_NAMES = [
    'bundle',
    'cover_sheaf',
    'diag_sheaf',
    'federated',
    'flat_bundle',
    'neural_diag_sheaf',
    'optimal_transport',
    'personalized_federated',
    'single_agent',
    'vanilla',
]

# Fixed per-agent color palette — consistent across every figure in the session
AGENT_COLORS = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_local_data(orch_dir: Path) -> dict[int, dict]:
    """Return ``{agent_idx: {'embs': Tensor, 'pos': Tensor}}`` from saved files."""
    result: dict[int, dict] = {}
    for f in sorted(orch_dir.glob('local_agent_*.pt')):
        idx = int(f.stem.rsplit('_', 1)[-1])
        result[idx] = torch.load(f, map_location='cpu', weights_only=False)
    return result


def _load_shared_data(orch_dir: Path) -> dict[tuple[str, str], dict]:
    """Return ``{(i_str, j_str): {'embs_i': Tensor, 'embs_j': Tensor}}``."""
    result: dict[tuple[str, str], dict] = {}
    for f in sorted(orch_dir.glob('shared_*.pt')):
        parts = f.stem.split('_')   # 'shared_i_j' → ['shared', 'i', 'j']
        edge = (parts[1], parts[2])
        result[edge] = torch.load(f, map_location='cpu', weights_only=False)
    return result


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _apply_edge_map(
    orchestrator,
    orch_name: str,
    edge: tuple[str, str],
    embs_i: torch.Tensor,
    embs_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the learned edge map and return ``(aligned_i, aligned_j)``.

    All returned tensors are detached and on CPU.

    Alignment conventions (what the training loss drives toward):

    ``bundle``
        Loss ``‖emb_i − emb_j @ O_ij.T‖`` → apply ``O_ij.T`` to j; i unchanged.

    ``diag_sheaf``
        Loss ``‖emb_i − emb_j @ D_ij‖`` (D diagonal so D.T = D) → apply D to j;
        i unchanged.

    ``cover_sheaf`` / ``vanilla`` / ``single_agent``
        No learned map → both returned unchanged.

    ``neural_diag_sheaf``
        Loss ``‖D_i(emb_i) − D_j(emb_j)‖`` → push both through their layers.

    ``optimal_transport``
        Loss ``‖T_i(emb_i) − T_j(emb_j)‖`` → push both through their layers.
    """
    sorted_edge = tuple(sorted(edge))  # maps are always keyed by sorted tuples

    if orch_name == 'bundle':
        O_ij = orchestrator.orthogonal_maps[sorted_edge].cpu()
        # emb_j @ O_ij.T ≈ emb_i  (j aligned to i's coordinate frame)
        return embs_i, embs_j @ O_ij.T

    elif orch_name == 'diag_sheaf':
        D_ij = orchestrator.diagonal_maps[sorted_edge].cpu()
        # D is diagonal → D.T == D; emb_j @ D_ij ≈ emb_i
        return embs_i, embs_j @ D_ij

    elif orch_name == 'neural_diag_sheaf':
        key = f'{sorted_edge[0]}_{sorted_edge[1]}'
        layer_i = orchestrator.diagonal_layers[key][sorted_edge[0]].cpu()
        layer_j = orchestrator.diagonal_layers[key][sorted_edge[1]].cpu()
        # D_i(emb_i) ≈ D_j(emb_j)  in the common transport space
        return layer_i(embs_i).detach(), layer_j(embs_j).detach()

    elif orch_name == 'optimal_transport':
        key = f'{sorted_edge[0]}_{sorted_edge[1]}'
        layer_i = orchestrator.transport_layers[key][sorted_edge[0]].cpu()
        layer_j = orchestrator.transport_layers[key][sorted_edge[1]].cpu()
        # T_i(emb_i) ≈ T_j(emb_j)  in the common transport space
        return layer_i(embs_i).detach(), layer_j(embs_j).detach()

    else:
        # cover_sheaf, vanilla, single_agent: identity (no learned edge map)
        return embs_i, embs_j


def _apply_reference_frame(
    orchestrator,
    orch_name: str,
    idx: int,
    embs: torch.Tensor,
) -> torch.Tensor:
    """Apply per-agent post-hoc transform for Group-1 orchestrators.

    For ``flat_bundle``:  ``embs @ R.T``  (R = ``local_reference_frames[idx]``)
    For federated variants: identity (no explicit transform available).
    """
    if orch_name == 'flat_bundle':
        R = orchestrator.local_reference_frames[str(idx)].cpu()
        return embs @ R.T
    return embs   # federated / personalized_federated


# ---------------------------------------------------------------------------
# Scatter helper
# ---------------------------------------------------------------------------

def _scatter(ax, embs: torch.Tensor, color, label: str) -> None:
    embs_np = embs.detach().cpu().numpy()
    ax.scatter(
        embs_np[:, 0], embs_np[:, 1],
        s=5, alpha=0.5, color=color, label=label, rasterized=True,
    )


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_group1(
    orchestrator,
    orch_name: str,
    agent_data: dict[int, dict],
    out_path: Path,
) -> None:
    """Two-panel figure: raw latents (left) vs. reference-frame-aligned (right).

    Both panels overlay all agents with consistent per-agent colors.
    For ``federated`` and ``personalized_federated`` the two panels are
    identical (no explicit post-hoc alignment map exists for those methods).
    """
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(16, 7))

    if orch_name == 'flat_bundle':
        titles = ('Private embeddings', 'Aligned embeddings')
    else:
        titles = ('Before alignment (raw)', 'After alignment')

    for idx, data in sorted(agent_data.items()):
        embs_raw = data['embs'].cpu()
        embs_aligned = _apply_reference_frame(orchestrator, orch_name, idx, embs_raw)
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        label = r'$b_{' + str(idx + 1) + r'}$'

        _scatter(ax_pre, embs_raw, color, label=label)
        _scatter(ax_post, embs_aligned, color, label=label)

    for ax, title in zip((ax_pre, ax_post), titles):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal', 'box')
        ax.legend(markerscale=3)
        ax.set_title(title, pad=12)

    fig.tight_layout(pad=1.0, w_pad=0.3)
    fig.subplots_adjust(wspace=-0.35)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_group2(
    orchestrator,
    orch_name: str,
    shared_data: dict[tuple[str, str], dict],
    out_path: Path,
) -> None:
    """Grid of before/after plots for every edge.

    Layout: ``n_edges`` rows × 2 columns.
    Left column  = raw shared embeddings (no map applied).
    Right column = embeddings after applying the learned edge alignment map.
    """
    edges = sorted(shared_data.keys())
    n_edges = len(edges)

    fig, axes = plt.subplots(
        n_edges, 2,
        figsize=(12, 6 * n_edges),
        squeeze=False,
    )

    for row, edge in enumerate(edges):
        i_str, j_str = edge
        i_idx, j_idx = int(i_str), int(j_str)
        color_i = AGENT_COLORS[i_idx % len(AGENT_COLORS)]
        color_j = AGENT_COLORS[j_idx % len(AGENT_COLORS)]

        embs_i = shared_data[edge]['embs_i'].cpu()
        embs_j = shared_data[edge]['embs_j'].cpu()

        label_i = r'$b_{' + str(i_idx + 1) + r'}$'
        label_j = r'$b_{' + str(j_idx + 1) + r'}$'

        # --- left: raw ---
        ax_pre = axes[row, 0]
        _scatter(ax_pre, embs_i, color_i, label_i)
        _scatter(ax_pre, embs_j, color_j, label_j)
        ax_pre.set_title(f'Edge ({i_idx + 1},{j_idx + 1}) — before alignment')
        ax_pre.set_xlabel('')
        ax_pre.set_ylabel('')
        ax_pre.set_aspect('equal', 'box')
        ax_pre.legend(markerscale=3)

        # --- right: aligned ---
        ax_post = axes[row, 1]
        t_i, t_j = _apply_edge_map(orchestrator, orch_name, edge, embs_i, embs_j)
        _scatter(ax_post, t_i, color_i, label_i)
        _scatter(ax_post, t_j, color_j, label_j)
        ax_post.set_title(f'Edge ({i_idx + 1},{j_idx + 1}) — after alignment')
        ax_post.set_xlabel('')
        ax_post.set_ylabel('')
        ax_post.set_aspect('equal', 'box')
        ax_post.legend(markerscale=3)

    fig.tight_layout(pad=1.0, h_pad=0.3, w_pad=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path='../config/hydra/',
    config_name='train',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    CURRENT = Path('.')
    ckpt_dir = get_checkpoint_dir(cfg, CURRENT)
    results_subdir = get_results_dir(cfg, CURRENT)
    project_name = cfg.logger.project

    for orch_name in ORCHESTRATOR_NAMES:
        orch_dir = results_subdir / orch_name
        if not orch_dir.exists():
            print(f'[{orch_name}] No results directory — skipping.')
            continue

        # ---- load checkpoint (needed for edge maps / reference frames) ----
        ckpt_path = find_best_checkpoint(ckpt_dir, project_name, orch_name)
        if ckpt_path is None:
            print(f'[{orch_name}] No checkpoint found — skipping.')
            continue

        print(f'\n[{orch_name}] Loading {ckpt_path.name}')
        orchestrator = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        orchestrator.eval()

        # ---- load saved latent representations ----
        agent_data = _load_local_data(orch_dir)
        if not agent_data:
            print(f'[{orch_name}] No local_agent_*.pt files found — skipping.')
            continue

        # ---- visualize ----
        if orch_name in GROUP1_ORCHS:
            out_path = orch_dir / 'latent_combined.pdf'
            _plot_group1(orchestrator, orch_name, agent_data, out_path)
            print(f'[{orch_name}] Saved → {out_path}')

        else:
            shared_data = _load_shared_data(orch_dir)
            if not shared_data:
                print(f'[{orch_name}] No shared_*.pt files found — skipping edge plots.')
                continue

            out_path = orch_dir / 'latent_edges.pdf'
            _plot_group2(orchestrator, orch_name, shared_data, out_path)
            print(f'[{orch_name}] Saved → {out_path}')


if __name__ == '__main__':
    main()
