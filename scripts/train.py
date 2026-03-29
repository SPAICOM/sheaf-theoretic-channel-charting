# train.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src import DeepMimoDataModule, DICHASUSDataModule


@hydra.main(
    config_path='../config/hydra/',
    config_name='train',
    version_base='1.3',
)
def main(cfg: DictConfig):
    # -----------------------------
    # Load dataset
    # -----------------------------
    if cfg.dataset.get('type') == 'dichasus':
        dm = DICHASUSDataModule(cfg.dataset, seed=cfg.seed)
    else:
        dm = DeepMimoDataModule(cfg.dataset, seed=cfg.seed)
    dm.prepare_data()
    dm.setup('fit')

    dm.train_dataloader()

    # -----------------------------
    # Model
    # -----------------------------
    model = instantiate(cfg.model, in_dim=dm.feature_dim)

    wandb_logger = WandbLogger(
        project=cfg.logger.project,
    )

    # # -----------------------------
    # # Trainer
    # # -----------------------------
    trainer = L.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=dm)

    return None


if __name__ == '__main__':
    main()
