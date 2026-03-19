""""""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# from src.agents import Agent
from src.datamodule import CSIDataModule
from src.utils import remove_non_empty_dir


# TODO Config yaml of hydra
@hydra.main(
    config_path='../config/hydra/',
    config_name='simulation',
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

    # Define some variables
    uuid: str = '...'

    # ===================================================
    #                  Wandb Logger
    # ===================================================
    # Convert DictConfig to a standard dictionary before passing to wandb
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # W&B login and Logger intialization
    wandb.login()
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=uuid,
        config=wandb_config,
        log_model=cfg.wandb.log_model,
    )

    # ===================================================
    #             Define the Trainer
    # ===================================================
    # # Callbacks definition
    callbacks = [
        LearningRateMonitor(logging_interval='step', log_momentum=True),
        ModelCheckpoint(monitor='valid/loss_epoch', save_top_k=1, mode='min'),
        BatchSizeFinder(mode='binsearch', max_trials=8),
        EarlyStopping(monitor='valid/loss_epoch', patience=10),
    ]

    # Initialiaze thr Trainer
    trainer: Trainer = Trainer(
        max_epochs=cfg.trainer.epochs,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        logger=wandb_logger,
        deterministic=cfg.trainer.deterministic,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # ===================================================
    #             Define the DataModule
    # ===================================================
    datamodule: CSIDataModule = CSIDataModule(
        dataset_cfg=cfg.dataset,
        seed=cfg.seed,
    )

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup('fit')

    # ===================================================
    #                Define the Agents
    # ===================================================
    # agents: list[Agent] = ...

    # ===================================================
    #                Define the Orchestrator
    # ===================================================
    # TODO create network edges dictionary:
    # {
    #   'agent_idx': {neighbors idx}
    # }
    orchestrator = ...

    # ===================================================
    #               Orchestrator Training
    # ===================================================
    trainer.fit(orchestrator, datamodule=datamodule)

    # Closing W&B
    wandb.finish()

    # Cleaning the working space
    remove_non_empty_dir('./wandb/')
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.wandb.project)

    return None


if __name__ == '__main__':
    main()
