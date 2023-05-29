import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.variational_dropout import VDLSTM


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor):
        lstm_out, hidden = self.lstm(x)
        return lstm_out, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


class VDEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, p: float):
        super(VDEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = VDLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropouto=p, batch_first=True)

    def forward(self, x: torch.Tensor):
        x, hidden = self.lstm(x)
        return x, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


class MLPDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_channels: int, output_steps: int, p: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.output_steps = output_steps
        self.p = p

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_channels * output_steps)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.reshape(-1, self.output_steps, self.num_channels)


class VDDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, p: float):
        super().__init__()
        self.lstm = VDLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropouto=p, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor):
        lstm_out, self.hidden = self.lstm(x.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(1))
        return output, self.hidden


class VDEncoderDecoder(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, output_steps: int, p: float, learning_rate: float = 0.01):
        super().__init__()
        self.enc_in_features = in_features
        self.enc_out_features = hidden_size
        self.output_steps = output_steps
        self.p = p

        self.encoder = VDEncoder(self.enc_in_features, hidden_size, 2, self.p)
        self.decoder = MLPDecoder(
            self.enc_out_features + self.enc_in_features, hidden_size, self.enc_in_features, self.output_steps, self.p
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def learn(self, train_loader: DataLoader, n_epochs: int) -> None:
        for epoch in range(n_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
                epoch_loss = 0
                for t, (x, y) in enumerate(tepoch):
                    # encoder outputs / decoder inputs
                    enc_output, _ = self.encoder(x)
                    decoder_input = torch.concat([x[:, -1, :], enc_output[:, -1, :]], dim=1)
                    # decoder forward
                    outputs = self.decoder(decoder_input)
                    # compute the loss
                    loss = F.smooth_l1_loss(outputs, y)

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss = (epoch_loss * t + loss.item()) / (t + 1)

                    # progress bar
                    tepoch.set_postfix(epoch_loss=epoch_loss)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        # encode input_tensor
        enc_output, _ = self.encoder(x)
        # decode input_tensor
        decoder_input = torch.concat([x[:, -1, :], enc_output[:, -1, :]], dim=1)
        outputs = self.decoder(decoder_input)
        outputs = outputs.cpu().numpy()
        return outputs


class VDSeq2Seq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, p: float = 0.1, learning_rate: float = 0.01):
        super().__init__()
        self.encoder = VDEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=2, p=p)
        self.decoder = VDDecoder(input_size=input_size, hidden_size=hidden_size, num_layers=2, p=p)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def learn(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        target_len: int,
        training_prediction: str = "recursive",
        teacher_forcing_ratio: float = 0.5,
        dynamic_tf: bool = False,
    ) -> np.ndarray:
        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        for epoch in range(n_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
                epoch_loss = 0
                for t_, (x, y) in enumerate(tepoch):
                    batch_loss = []
                    # outputs tensor
                    outputs = torch.zeros(x.shape[0], target_len, x.shape[2])
                    # zero the gradient
                    self.optimizer.zero_grad()
                    # encoder outputs
                    _, encoder_hidden = self.encoder(x)
                    # decoder with teacher forcing
                    decoder_input = x[:, -1, :]  # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == "recursive":
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[:, t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == "teacher_forcing":
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[:, t] = decoder_output
                                decoder_input = y[:, t]

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[:, t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == "mixed_teacher_forcing":
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[:, t] = decoder_output

                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = y[:, t]
                            # predict recursively
                            else:
                                decoder_input = decoder_output

                    # compute the loss
                    loss = F.smooth_l1_loss(outputs, y)
                    batch_loss.append(loss.item())

                    # backpropagation
                    loss.backward()
                    self.optimizer.step()

                    # progress bar
                    epoch_loss = (epoch_loss * t_ + loss.item()) / (t_ + 1)
                    tepoch.set_postfix(epoch_loss=epoch_loss)

                # loss for epoch
                losses[epoch] = np.mean(batch_loss)

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

        return losses

    @torch.no_grad()
    def predict(self, x: torch.Tensor, target_len: int) -> np.ndarray:
        # encode input_tensor
        _, encoder_hidden = self.encoder(x)
        # initialize tensor for predictions
        outputs = torch.zeros(x.shape[0], target_len, x.shape[2])
        # decode input_tensor
        decoder_input = x[:, -1, :]
        decoder_hidden = encoder_hidden
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output
            decoder_input = decoder_output
        outputs = outputs.cpu().numpy()
        return outputs
