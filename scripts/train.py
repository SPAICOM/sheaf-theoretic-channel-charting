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

    loader = dm.train_dataloader()
    print('LOCAL DATASETS')
    for k, ds in dm.train_local_dataset.items():
        print(k, len(ds))

    print('\nSHARED DATASETS')
    for k, ds in dm.train_shared_dataset.items():
        print(k, len(ds))

    batch = next(iter(loader))
    example_agent = batch[0][0]
    print(
        len(example_agent),
        example_agent[0].shape,
        example_agent[1].shape,
        example_agent[2].shape,
        example_agent[2].shape,
    )

    # batch = next(iter(dm.train_dataloader()))
    # print(batch)

    # xA, xP, xN, y = batch
    # print(xA.shape, xP.shape, None if xN is None else xN.shape, y)
    # print(xA)

    # -----------------------------
    # Model
    # -----------------------------
    # model = instantiate(cfg.model, in_dim=dm.feature_dim)

    # wandb_logger = WandbLogger(
    #     project=cfg.logger.project,
    # )

    # # -----------------------------
    # # Trainer
    # # -----------------------------
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
