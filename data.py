"""Data loading for MDLM D-Head Accelerator.

Uses the pre-tokenized OpenWebText cache from the MDLM repo.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import datasets

from config import Config


class ArrowDataset(Dataset):
    """Thin wrapper around HuggingFace Arrow dataset."""

    def __init__(self, hf_dataset, seq_len: int):
        self.ds = hf_dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ids = self.ds[idx]['input_ids']
        return {'input_ids': torch.tensor(ids[:self.seq_len], dtype=torch.long)}


def get_dataloader(
    config: Config,
    split: str = "train",
) -> DataLoader:
    """Load pre-tokenized OpenWebText from MDLM cache."""
    path = config.train_data if split == "train" else config.valid_data
    print(f"Loading cached dataset from {path}")
    hf_ds = datasets.load_from_disk(path)
    dataset = ArrowDataset(hf_ds, config.max_length)
    print(f"Dataset: {len(dataset)} sequences of length {config.max_length}")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader
