import lightning as L
import torch
import torch.nn as nn


class BaseOrchestrator(L.LightningModule):
    def __init__(
        self,
        agents: list[nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        lmb: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Agents list
        self.hparams['agents'] = nn.ModuleList(agents)

    def on_train_epoch_end(self):
        pass

    def forward(
        self,
        batch: dict[str, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        pass

    def _shared_eval(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """A common step performed in the test and validation step.

        Args:
            batch : dict[str, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (output, total_loss) : tuple[dict[int, torch.Tensor], torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        pass

    def training_step(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch : dict[str, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        _, loss = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='train',
        )
        return loss

    def test_step(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """The test step.

        Args:
            batch : dict[str, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='test',
        )
        return None

    def validation_step(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """The validation step.

        Args:
            batch : dict[str, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            output : dict[int, torch.Tensor]
                The output of the network.
        """
        output, _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='validation',
        )
        return output

    def predict_step(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """The predict step.

        Args:
            batch : dict[str, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.

        Returns:
            dict[int, torch.Tensor]
                The output of the network.
        """
        return self(batch)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure the optimizer used for training.

        Uses the AdamW optimizer with the learning rate defined in
        ``self.hparams.LR``.

        Returns:
            dict[str, object]: A dictionary containing the optimizer used by
            the training loop. The dictionary has the following key:

            - "optimizer": The instantiated AdamW optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.LR)
        return {
            'optimizer': optimizer,
        }
