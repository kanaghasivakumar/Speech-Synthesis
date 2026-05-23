import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LibriTTSDataset(Dataset):
    def __init__(self, manifest_path, preprocessed_dir, split="train"):
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        with open(manifest_path) as f:
            self.items = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        uid = self.items[idx]
        base = os.path.join(self.preprocessed_dir, uid)

        phonemes = np.load(f"{base}_phonemes.npy")
        mel = np.load(f"{base}_mel.npy")
        pitch = np.load(f"{base}_pitch.npy")
        energy = np.load(f"{base}_energy.npy")
        duration = np.load(f"{base}_duration.npy")

        return {
            "uid": uid,
            "phonemes": torch.tensor(phonemes, dtype=torch.long),
            "mel": torch.tensor(mel, dtype=torch.float32),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "duration": torch.tensor(duration, dtype=torch.long),
            "src_len": len(phonemes),
            "mel_len": mel.shape[0],
        }