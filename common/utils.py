import glob

import numpy as np
import pandas as pd


def load_channels(root_folder: str, n_channels: int = 3, handle_outliers: bool = True) -> list[pd.DataFrame]:
    filenames = sorted(glob.glob(root_folder))
    dataset = []
    for i, path in enumerate(filenames[:n_channels]):
        data = pd.read_csv(path, sep='\t', names=['date', 'q', 'ptx', 'gamma', 'pdl'])
        time_idx = np.arange(len(data))
        group = np.full(len(data), i)
        data['group'] = group
        data['time_idx'] = time_idx
        dataset.append(data)
    if handle_outliers:
        for i in range(n_channels):
            q_factor = dataset[i]['q']
            q10, q90 = q_factor.quantile([0.10, 0.90])
            iqr = q90 - q10
            cutoff = iqr * 1.5
            lower, upper = q10 - cutoff, q90 + cutoff
            dataset[i].loc[(q_factor < lower) + (q_factor > upper), 'q'] = np.nan
            dataset[i]['q'].interpolate(method='linear', inplace=True)
    return dataset

def train_test_val_split(dataset: list[pd.DataFrame], train_size: float = 0.6, val_size: float = 0.2) -> tuple[pd.DataFrame]:
    train_dataset = []
    val_dataset = []
    test_dataset = []
    train_samples = int(36160 * train_size)
    val_samples = int(36160 * val_size)
    for data in dataset:
        train_dataset.append(data.iloc[:train_samples].copy())
        val_dataset.append(data.iloc[train_samples:train_samples+val_samples].copy())
        test_dataset.append(data.iloc[train_samples+val_samples:].copy())
    train_df = pd.concat(train_dataset, ignore_index=True)
    val_df = pd.concat(val_dataset, ignore_index=True)
    test_df = pd.concat(test_dataset, ignore_index=True)
    return train_df, val_df, test_df


