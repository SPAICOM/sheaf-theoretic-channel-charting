""""""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from collections import defaultdict

import hydra
from hydra.utils import instantiate

# import omegaconf
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.datamodule import CSIDataModule
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
    #                  Wandb Logger
    # ===================================================
    logger = instantiate(cfg.logger)

    # Log full Hydra config to WandB
    if logger is not None:
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
        )

    # ===================================================
    #             Define the Trainer
    # ===================================================
    # Instantiate callbacks
    if cfg.callbacks is None:
        callbacks = []
    else:
        callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    # Instantiate Trainer
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # ===================================================
    #             Define the DataModule
    # ===================================================
    datamodule = CSIDataModule(cfg.dataset, seed=cfg.seed)
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

    # -------------------------
    # Train
    # -------------------------
    trainer.fit(orchestrator, datamodule=datamodule)

    # -------------------------
    # Test
    # -------------------------
    orchestrator.eval_all(K_max=10)

    # Cleaning the working space
    remove_non_empty_dir('./wandb/')
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)

    return None


if __name__ == '__main__':
    main()
