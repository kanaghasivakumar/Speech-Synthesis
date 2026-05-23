import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    batch.sort(key=lambda x: x["src_len"], reverse=True)

    uids = [b["uid"] for b in batch]
    src_lens = torch.tensor([b["src_len"] for b in batch])
    mel_lens = torch.tensor([b["mel_len"] for b in batch])

    phonemes = pad_sequence([b["phonemes"] for b in batch], batch_first=True, padding_value=0)
    mel = pad_sequence([b["mel"] for b in batch], batch_first=True, padding_value=0.0)
    pitch = pad_sequence([b["pitch"] for b in batch], batch_first=True, padding_value=0.0)
    energy = pad_sequence([b["energy"] for b in batch], batch_first=True, padding_value=0.0)
    duration = pad_sequence([b["duration"] for b in batch], batch_first=True, padding_value=0)

    return {
        "uids": uids,
        "phonemes": phonemes,
        "mel": mel,
        "pitch": pitch,
        "energy": energy,
        "duration": duration,
        "src_lens": src_lens,
        "mel_lens": mel_lens,
    }