import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class UnivariateTimeSeriesMLP(nn.Module):
    def __init__(self, input_size: int, target_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.Linear(512, target_size))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def learn(self, train_loader: DataLoader, n_epochs: int) -> None:
        for epoch in range(n_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
                epoch_loss = 0.0
                for t, (x, y) in enumerate(tepoch):
                    # very ugly hack for reusing the same dataloader as the lstm
                    # basically we need to reshape the input tensors to be:
                    # [batch_size, seq_len, input_size] -> [batch_size, input_size, seq_len]
                    # in this way it automatically batches over all the time-series,
                    # fundamentally processing each sequence independently from the other
                    # the fact that it requires six paragraphs to explain is a telling sign
                    x = torch.permute(x, (0, 2, 1))
                    y = torch.permute(y, (0, 2, 1))
                    # forward pass
                    outputs = self.forward(x)
                    loss = F.mse_loss(outputs, y)

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss = (epoch_loss * t + loss.item()) / (t + 1)

                    # progress bar
                    tepoch.set_postfix(epoch_loss=epoch_loss)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> npt.NDArray:
        # same trick as above
        x = torch.permute(x, (0, 2, 1))
        # decode input_tensor
        outputs = self.layers(x).numpy()
        return outputs


class MultiVariateTimeSeriesMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_channels: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.layers = nn.Sequential(
            nn.Linear(n_channels * input_size, 512), nn.ReLU(), nn.Linear(512, n_channels * output_size)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape the output properly to match
        # the shape of the target sequences
        return self.layers(x)

    def learn(self, train_loader: DataLoader, n_epochs: int) -> None:
        for epoch in range(n_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
                epoch_loss = 0.0
                for t, (x, y) in enumerate(tepoch):
                    # very ugly hack (again) for reusing the same dataloader as the lstm
                    # this time we flatten the arrays to have shape:
                    # [batch_size, seq_len, input_size] -> [batch_size, input_size * seq_len]
                    # in this way we batch only over the actual batch, but we process all the
                    # time-series jointly.
                    x = torch.flatten(x, start_dim=1)
                    y = torch.flatten(y, start_dim=1)
                    # forward pass
                    outputs = self.forward(x)
                    loss = F.mse_loss(outputs, y)

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss = (epoch_loss * t + loss.item()) / (t + 1)

                    # progress bar
                    tepoch.set_postfix(epoch_loss=epoch_loss)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> npt.NDArray:
        # same trick as above
        x = torch.flatten(x, start_dim=1)
        # decode input_tensor
        outputs = self.layers(x).numpy()
        return outputs
