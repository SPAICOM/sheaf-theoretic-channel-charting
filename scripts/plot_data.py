# train.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from src import CSIDataModule


@hydra.main(
    config_path='../config/hydra/',
    config_name='train',
    version_base='1.3',
)
def main(cfg: DictConfig):
    # -----------------------------
    # Load DeepMIMO
    # -----------------------------
    dm = CSIDataModule(cfg.dataset, seed=cfg.seed)
    dm.prepare_data()
    dm.setup('fit')

    # dm.train_dataloader()
    dm.plot_trajectories(n=1, n_clusters=4, sequential_shading=True)
    dm.plot_trajectories(n=1, n_clusters=4, sequential_shading=True, stage='test')
    dm.plot_trajectories(n=1, n_clusters=4, sequential_shading=True, stage='val')

    return None


if __name__ == '__main__':
    main()
