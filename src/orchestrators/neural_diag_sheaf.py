"""Neural Diagonal Sheaf Channel Charting Orchestrator.

This orchestrator implements a neural diagonal sheaf approach where each edge
maintains learnable diagonal (element-wise scaling) transformations that are
trained via gradient descent rather than computed in closed form.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DiagonalTransportLayer(nn.Module):
    """Neural network layer for diagonal (element-wise) transport.

    Applies a learnable element-wise scaling transformation to input embeddings.
    Unlike the closed-form diagonal sheaf, this is learned via backpropagation.

    Parameters
    ----------
    in_dim : int, optional
        Input dimension (embedding size). Default: 2.
    """

    def __init__(
        self,
        in_dim: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim

        # Learnable diagonal scaling parameters (one per dimension)
        self.D = nn.Parameter(torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply diagonal transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape (batch_size, in_dim).
        """
        return x * self.D


class NeuralDiagSheafCC(BaseOrchestrator):
    """Neural Diagonal Sheaf Channel Charting orchestrator.

    Implements alignment via learnable diagonal linear maps that are trained
    via gradient descent (unlike DiagSheafCC which solves in closed form).
    Each edge maintains two diagonal transformation layers, one for each
    base station endpoint.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network models.
    neighbors : dict[int, set[int]]
        Dictionary mapping agent indices to sets of neighboring agent indices.
    lr : float
        Learning rate for optimization.
    weight_decay : float
        L2 regularization strength.
    lmb_min : float, optional
        Minimum value for alignment loss weight. Default: 1e-3.
    lmb_max : float, optional
        Maximum value for alignment loss weight. Default: 1.0.
    lmb_schedule : str, optional
        Schedule for interpolation between lmb_min and lmb_max. Options:
        'linear', 'exponential', 'cosine'. Default: 'cosine'.
    transition_epoch : float | None, optional
        Epoch at which to transition from lmb_min to lmb_max. Default: None.
    steepness : float, optional
        Steepness of sigmoid transition. Default: 0.1.
    n_clusters : int | None, optional
        Number of clusters for latent space visualization. Default: None.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float,
        lmb_min: float = 1e-3,
        lmb_max: float = 1.0,
        lmb_schedule: str = 'cosine',
        transition_epoch: float | None = None,
        steepness: float = 0.1,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            weight_decay=weight_decay,
            lr=lr,
            transition_epoch=transition_epoch,
            steepness=steepness,
            n_clusters=n_clusters,
        )
        self._lmb = lmb_min
        self.save_hyperparameters()

        # Build edge list from neighbor graph (undirected, sorted tuples)
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Create learnable diagonal transport layers for each edge
        # Each edge has two layers: one for each endpoint base station
        self.diagonal_layers = nn.ModuleDict(
            {
                f'{i}_{j}': nn.ModuleDict(
                    {
                        i: DiagonalTransportLayer(self.agents['0'].out_dim),
                        j: DiagonalTransportLayer(self.agents['0'].out_dim),
                    }
                )
                for (i, j) in self.hparams['edges']
            }
        )

    def on_train_epoch_start(self) -> None:
        """Update lmb according to the schedule at the start of each epoch.

        Computes the alignment loss weight (lmb) based on the configured schedule
        (linear, exponential, or cosine) and current epoch progress.
        """
        max_epochs = self.trainer.max_epochs
        t = self.current_epoch / max(max_epochs - 1, 1)

        lmb_min = self.hparams['lmb_min']
        lmb_max = self.hparams['lmb_max']
        schedule = self.hparams['lmb_schedule']

        match schedule:
            case 'linear':
                self._lmb = lmb_min + (lmb_max - lmb_min) * t
            case 'exponential':
                self._lmb = lmb_min * (lmb_max / lmb_min) ** t
            case 'cosine':
                self._lmb = lmb_min + (lmb_max - lmb_min) * (1 - math.cos(math.pi * t)) / 2
            case _:
                raise ValueError(
                    f"Unknown lmb_schedule: '{schedule}'. Choose from: linear, exponential, cosine."
                )

        self.log('train/lmb', self._lmb, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        """Plot latent space at epoch end.

        Visualizes the learned embeddings after each training epoch.
        """
        self.plot_latent_space(
            split='train',
            prefix='trajectory_train',
            last_epoch_only=True,
            n_clusters=self.hparams.n_clusters,
        )

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """A common step performed in the test and validation step.

        Computes private losses for each agent's local data and transport losses
        for shared data between neighboring agents using learnable diagonal layers.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            The current batch containing both private and shared data.
        batch_idx : int
            The batch index.
        prefix : str
            The step type for logging purposes ('train', 'val', 'test').

        Returns
        -------
        tuple[dict[int, torch.Tensor], torch.Tensor]
            Tuple of (private outputs dict, total loss).
        """
        private_outputs = self(batch)
        on_step = prefix == 'train'

        # Compute embeddings of the overlapping areas between base stations
        shared_outputs = {
            (i, j): {
                i: self.agents[i](
                    batch[(int(i), int(j))][0],
                    triplet_mode=False,
                ),
                j: self.agents[j](
                    batch[(int(i), int(j))][1],
                    triplet_mode=False,
                ),
            }
            for (i, j) in self.hparams['edges']
        }

        total_private_loss = 0.0
        total_transport_loss = 0.0

        # Compute the personalized loss for each agent on its private data
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            agent_losses = agent.compute_loss(batch[int(idx)], private_outputs[int(idx)])

            for loss_name, loss_val in agent_losses.items():
                self.log(
                    f'{prefix}/{loss_name}_agent_{idx}',
                    loss_val,
                    on_step=on_step,
                    on_epoch=True,
                    batch_size=batch_size,
                    prog_bar=False,
                )

            # Alpha-weighted combination of reconstruction and triplet loss
            if 'rec_loss' in agent_losses:
                alpha = self._compute_alpha(self.current_epoch)
                for loss_name, loss_val in agent_losses.items():
                    match loss_name:
                        case 'rec_loss':  # Weight = (1 - alpha)
                            total_private_loss += (1 - alpha) * loss_val
                        case 'triplet_loss':  # Weight = alpha
                            total_private_loss += alpha * loss_val
                        case _:  # Other losses - full weight
                            total_private_loss += loss_val
            else:
                # No reconstruction loss - use full triplet loss
                total_private_loss += sum(agent_losses.values())

        # Compute transport losses using learnable diagonal transformations
        # Apply each endpoint's diagonal layer and measure alignment distance
        for i, j in self.hparams['edges']:
            transport_i = self.diagonal_layers[f'{i}_{j}'][str(i)](shared_outputs[(i, j)][i])
            transport_j = self.diagonal_layers[f'{i}_{j}'][str(j)](shared_outputs[(i, j)][j])

            # Transport loss: mean squared distance after diagonal transformation
            transport_loss = (torch.linalg.norm(transport_i - transport_j, dim=1) ** 2).mean()

            total_transport_loss += transport_loss

        # Total loss = private loss + lambda * transport loss
        total_loss = total_private_loss + self._lmb * total_transport_loss

        self.log_dict(
            {
                f'{prefix}/total_private_loss': total_private_loss,
                f'{prefix}/total_alignment_loss': total_transport_loss,
                f'{prefix}/total_loss': total_loss,
            },
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return private_outputs, total_loss

    def _compute_FOSCTTM(self, split: str = 'train') -> torch.Tensor:
        """Compute FOSCTTM (Fraction Of Successive Correct Triplet Matches) metric.

        Evaluates the quality of alignment by measuring how often the nearest neighbor
        in the aligned embedding space correctly matches the ground truth correspondence.

        Parameters
        ----------
        split : str, optional
            Which dataset split to use ('train' or 'test'). Default: 'train'.

        Returns
        -------
        torch.Tensor
            Mean FOSCTTM score across all edges.
        """
        shared_dataset = (
            self.trainer.datamodule.train_shared_dataset
            if split == 'train'
            else self.trainer.datamodule.test_shared_dataset
        )
        FOSCTTM = torch.zeros(len(self.hparams['edges']))

        for i, dataset in enumerate(shared_dataset.values()):
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            bs1_str = str(dataset.idx_bs_1)
            bs2_str = str(dataset.idx_bs_2)
            node_i, node_j = tuple(sorted((bs1_str, bs2_str)))
            
            # Accumulate embeddings over the full shared dataset
            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            # Stack all embeddings: (N, d)
            Z_i = torch.cat(embs_1, dim=0)
            Z_j = torch.cat(embs_2, dim=0)

            # Perform edge alignment using learned diagonal layers
            Z_i_hat = self.diagonal_layers[f'{node_i}_{node_j}'][bs1_str](Z_i)
            Z_j_hat = self.diagonal_layers[f'{node_i}_{node_j}'][bs2_str](Z_j)
            edge_FOSCTTM = torch.zeros(Z_j_hat.shape[0])

            # Point-wise FOSCTTM: for each point, measure fraction of neighbors
            # where the nearest neighbor correctly identifies the correspondence
            for p in range(Z_j_hat.shape[0]):
                d = torch.linalg.norm(Z_j_hat[p, :] - Z_i_hat[p, :])

                # Distance from point p in j to all points in i
                Ds_1 = torch.linalg.norm(Z_j_hat[p, :] - Z_i_hat, dim=1)

                # Distance from point p in i to all points in j
                Ds_2 = torch.linalg.norm(Z_i_hat[p, :] - Z_j_hat, dim=1)

                # Fraction of points correctly identified as closer than the true match
                edge_FOSCTTM[p] = 0.5 * (
                    torch.sum(Ds_1 < d) / Z_j_hat.shape[0] + torch.sum(Ds_2 < d) / Z_j_hat.shape[0]
                )

            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
