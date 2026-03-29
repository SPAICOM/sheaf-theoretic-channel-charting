"""Single-agent DataModule that returns plain DataLoaders (no CombinedLoader).

Wraps DeepMimoDataModule and exposes only agent 0's local dataset via
standard PyTorch DataLoaders, bypassing the multi-agent CombinedLoader.
"""

from torch.utils.data import DataLoader

from .deepmimo import DeepMimoDataModule


class SingleAgentDataModule(DeepMimoDataModule):
    """DataModule for single-agent training on agent index 0.

    Overrides the dataloader methods of DeepMimoDataModule to return plain
    DataLoaders instead of CombinedLoader, using only the local dataset of
    the first base station (index 0).

    Parameters
    ----------
    dataset_cfg : DictConfig | dict
        Same configuration accepted by DeepMimoDataModule.
    seed : int | None
        Random seed forwarded to DeepMimoDataModule.
    """

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_local_dataset[0],
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cfg['pin_memory'],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_local_dataset[0],
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cfg['pin_memory'],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_local_dataset[0],
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cfg['pin_memory'],
        )
