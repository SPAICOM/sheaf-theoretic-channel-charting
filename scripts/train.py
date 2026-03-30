# train.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from src import DeepMimoDataModule, DICHASUSDataModule
from src.viz import plot_positions_2d


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

    # Visualize positions for DICHASUS dataset (disabled by default due to slow lazy loading)
    if cfg.dataset.get('type') == 'dichasus' and cfg.dataset.get('viz_enabled', False):
        imgs_dir = Path('imgs')
        imgs_dir.mkdir(exist_ok=True)

        # Configurable split: 'train', 'val', or 'test' (default: test)
        split = cfg.dataset.get('viz_split', 'test')

        plot_positions_2d(
            dm,
            split=split,
            per_bs=True,
            color_by_time=True,
            max_samples=100,
            save_path=str(imgs_dir / f'positions_{split}.png'),
            show=False,
        )
        print(f'Saved position plot to {imgs_dir / f"positions_{split}.png"}')

    print(next(iter(dm.train_dataloader())))

    # # -----------------------------
    # # Model
    # # -----------------------------
    # model = instantiate(cfg.model, in_dim=dm.feature_dim)

    # wandb_logger = WandbLogger(
    #     project=cfg.logger.project,
    # )

    # # # -----------------------------
    # # # Trainer
    # # # -----------------------------
    # trainer = L.Trainer(
    #     max_epochs=cfg.model.max_epochs,
    #     accelerator='auto',
    #     devices='auto',
    #     log_every_n_steps=10,
    #     logger=wandb_logger,
    # )

    # trainer.fit(model, datamodule=dm)

    return None


if __name__ == '__main__':
    main()
