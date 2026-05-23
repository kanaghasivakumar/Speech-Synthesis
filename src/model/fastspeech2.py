import torch
import torch.nn as nn
from src.model.encoder import Encoder
from src.model.variance_adaptor import VarianceAdaptor
from src.model.decoder import Decoder


class PostNet(nn.Module):
    def __init__(self, n_mels, channels=512, kernel_size=5, n_layers=5):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = n_mels if i == 0 else channels
            out_ch = n_mels if i == n_layers - 1 else channels
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
            ]
            if i < n_layers - 1:
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x.transpose(1, 2)).transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.variance_adaptor = VarianceAdaptor(cfg)
        self.decoder = Decoder(cfg)
        self.postnet = PostNet(cfg.audio.n_mels)

    def forward(self, phonemes, src_lens, mel_lens=None, max_mel_len=None,
                pitch=None, energy=None, duration=None,
                p_control=1.0, e_control=1.0, d_control=1.0):
        src_mask = self._make_pad_mask(phonemes, src_lens)
        enc_out, enc_attn = self.encoder(phonemes, src_mask)

        (va_out, log_dur_pred, pitch_pred, energy_pred,
         pred_mel_lens, _) = self.variance_adaptor(
            enc_out, src_mask, max_len=max_mel_len,
            pitch_target=pitch, energy_target=energy, duration_target=duration,
            p_control=p_control, e_control=e_control, d_control=d_control,
        )

        out_lens = mel_lens if mel_lens is not None else pred_mel_lens
        mel_mask = self._make_pad_mask(va_out, out_lens)
        mel_out, dec_attn = self.decoder(va_out, mel_mask)
        mel_postnet = self.postnet(mel_out)

        return mel_out, mel_postnet, log_dur_pred, pitch_pred, energy_pred, \
               src_mask, mel_mask, enc_attn, dec_attn

    @staticmethod
    def _make_pad_mask(x, lens):
        B, T = x.size(0), x.size(1)
        mask = torch.arange(T, device=lens.device).unsqueeze(0) >= lens.unsqueeze(1)
        return mask