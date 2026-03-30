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
    #                  Wandb Logger
    # ===================================================
    logger = instantiate(cfg.logger)

    if logger is not None:
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
        )

    # ===================================================
    #             Define the Trainer
    # ===================================================
    if cfg.callbacks is None:
        callbacks = []
    else:
        callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # ===================================================
    #             Define the DataModule
    # ===================================================
    datamodule = DICHASUSSingleAgentDataModule(cfg.dataset, seed=cfg.seed)
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

    # ===================================================
    #                     Train
    # ===================================================
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
