import torch
import torch.nn as nn
from src.modules.length_regulator import LengthRegulator, DurationPredictor, VariancePredictor


class VarianceAdaptor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.model.encoder_hidden
        fs = cfg.model.variance_predictor_filter_size
        ks = cfg.model.variance_predictor_kernel_size
        drop = cfg.model.variance_predictor_dropout
        n_bins = cfg.model.n_bins

        self.duration_predictor = DurationPredictor(d, fs, ks, drop)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d, fs, ks, drop)
        self.energy_predictor = VariancePredictor(d, fs, ks, drop)

        self.pitch_embed = nn.Embedding(n_bins + 1, d, padding_idx=0)
        self.energy_embed = nn.Embedding(n_bins + 1, d, padding_idx=0)
        self.register_buffer("pitch_bins", torch.zeros(n_bins))
        self.register_buffer("energy_bins", torch.zeros(n_bins))

    def set_pitch_bins(self, lo, hi, log_scale=True):
        if log_scale:
            self.pitch_bins = torch.exp(torch.linspace(
                torch.tensor(lo).log(), torch.tensor(hi).log(), self.pitch_bins.size(0)
            ))
        else:
            self.pitch_bins = torch.linspace(lo, hi, self.pitch_bins.size(0))

    def set_energy_bins(self, lo, hi):
        self.energy_bins = torch.linspace(lo, hi, self.energy_bins.size(0))

    def _bucketize(self, x, bins):
        return torch.bucketize(x, bins)

    def forward(self, x, src_mask, mel_mask=None, max_len=None,
                pitch_target=None, energy_target=None, duration_target=None,
                p_control=1.0, e_control=1.0, d_control=1.0):
        log_dur_pred = self.duration_predictor(x, src_mask)
        pitch_pred = self.pitch_predictor(x, src_mask)
        energy_pred = self.energy_predictor(x, src_mask)

        if duration_target is not None:
            x, mel_lens = self.length_regulator(x, duration_target, max_len)
            dur_rounded = duration_target
        else:
            dur_rounded = torch.clamp(
                torch.round(torch.exp(log_dur_pred) - 1) * d_control, min=0
            )
            x, mel_lens = self.length_regulator(x, dur_rounded, max_len)

        if pitch_target is not None:
            pitch_emb = self.pitch_embed(self._bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_emb = self.pitch_embed(self._bucketize(pitch_pred * p_control, self.pitch_bins))
        x = x + pitch_emb

        if energy_target is not None:
            energy_emb = self.energy_embed(self._bucketize(energy_target, self.energy_bins))
        else:
            energy_emb = self.energy_embed(self._bucketize(energy_pred * e_control, self.energy_bins))
        x = x + energy_emb

        return x, log_dur_pred, pitch_pred, energy_pred, mel_lens, mel_mask