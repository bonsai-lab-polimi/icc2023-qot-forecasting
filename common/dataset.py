from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, encoder_length, decoder_length) -> None:
        super().__init__()
        self.data = data
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        
    def __len__(self):
        return len(self.data) - self.encoder_length - self.decoder_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.encoder_length].astype('float32')
        y = self.data[idx+self.encoder_length:idx+self.encoder_length+self.decoder_length].astype('float32')
        return x, y
