import torch
import torch.nn.functional as F

# try pytorch lightning because: why not
from pytorch_lightning import LightningModule
from torch import nn


class SingleInputSingleOutputMLP(LightningModule):
    def __init__(self, input_size: int, target_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.Linear(512, target_size))

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        # batch_size, seq_len, input_size -> batch_size, input_size, seq_len
        # as such it batches over all the inputs in the batch, basically processing
        # each sequence independently from the other
        x = torch.permute(x, (0, 2, 1))
        y = torch.permute(y, (0, 2, 1))
        loss = F.mse_loss(self.forward(x), y)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        x = torch.permute(x, (0, 2, 1))
        y = torch.permute(y, (0, 2, 1))
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        x = torch.permute(x, (0, 2, 1))
        y = torch.permute(y, (0, 2, 1))
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        self.log("test_rmse", torch.sqrt(mse))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class MultiInputMultiOutputMLP(LightningModule):
    def __init__(self, input_size: int, output_size: int, n_channels: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.layers = nn.Sequential(
            nn.Linear(n_channels * input_size, 512), nn.ReLU(), nn.Linear(512, n_channels * output_size)
        )

    def forward(self, x: torch.Tensor):
        # reshape the output properly to match
        # the shape of the target sequences
        return self.layers(x).reshape(-1, self.output_size, self.n_channels)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        # flattens the input to consider jointly
        # all channels and all time steps
        x = torch.flatten(x, start_dim=1)
        loss = F.mse_loss(self.forward(x), y)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        x = torch.flatten(x, start_dim=1)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        x = torch.flatten(x, start_dim=1)
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        self.log("test_rmse", torch.sqrt(mse))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
