import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x.view(x.size(0), -1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
