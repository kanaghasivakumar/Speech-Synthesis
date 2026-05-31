import torch
import torch.nn as nn


class LengthRegulator(nn.Module):
    def forward(self, x, durations, max_len=None):
        outputs = []
        for xi, di in zip(x, durations):
            output = torch.repeat_interleave(xi, di.long().clamp(min=0), dim=0)
            outputs.append(output)
        out = self._pad(outputs, max_len)
        mel_lens = torch.tensor([o.size(0) for o in outputs], device=x.device)
        return out, mel_lens

    def _pad(self, seqs, max_len=None):
        T = max_len if max_len is not None else max(s.size(0) for s in seqs)
        d = seqs[0].size(1)
        out = seqs[0].new_zeros(len(seqs), T, d)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out


class DurationPredictor(nn.Module):
    def __init__(self, d_model, filter_size, kernel_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            Conv1dNorm(d_model, filter_size, kernel_size),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
            Conv1dNorm(filter_size, filter_size, kernel_size),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, x, mask=None):
        out = self.net(x)
        out = self.linear(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class VariancePredictor(nn.Module):
    def __init__(self, d_model, filter_size, kernel_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            Conv1dNorm(d_model, filter_size, kernel_size),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
            Conv1dNorm(filter_size, filter_size, kernel_size),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, x, mask=None):
        out = self.net(x)
        out = self.linear(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class Conv1dNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)