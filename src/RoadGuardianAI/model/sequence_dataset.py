import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional

class SequenceDataset(Dataset):
    """
    Build sliding-window sequence dataset by segment_id.
    Each sample: (seq_len, n_features) tensor -> label for last timestep.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        segment_col: str = "segment_id",
        ts_col: str = "ts",
        feature_cols: List[str] = None,
        target_col: str = "label_1h",
        seq_len: int = 12,
        stride: int = 1,
    ):
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.feature_cols = list(feature_cols) if feature_cols is not None else []
        self.target_col = target_col

        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values([segment_col, ts_col])
        self.samples = []

        grouped = df.groupby(segment_col, sort=False)
        for seg, g in grouped:
            if len(g) < self.seq_len:
                continue
            arr_x = g[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            arr_y = g[self.target_col].astype(int).to_numpy(dtype=np.int64)
            n = arr_x.shape[0]
            for start in range(0, n - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                x = arr_x[start:end]
                y = int(arr_y[end - 1])
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    xstack = torch.stack(xs)
    return xstack, ys
