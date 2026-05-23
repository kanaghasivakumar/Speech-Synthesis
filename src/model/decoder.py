import torch.nn as nn
from src.model.encoder import PositionalEncoding
from src.modules.fft_block import FFTBlock


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_enc = PositionalEncoding(cfg.model.decoder_hidden, cfg.model.decoder_dropout, cfg.model.max_seq_len)
        self.layers = nn.ModuleList([
            FFTBlock(
                cfg.model.decoder_hidden,
                cfg.model.decoder_head,
                cfg.model.fft_conv_filter_size,
                cfg.model.decoder_kernel_size,
                cfg.model.decoder_dropout,
            )
            for _ in range(cfg.model.decoder_layers)
        ])
        self.proj = nn.Linear(cfg.model.decoder_hidden, cfg.audio.n_mels)

    def forward(self, x, mel_mask):
        out = self.pos_enc(x)
        attn_weights = []
        for layer in self.layers:
            out, w = layer(out, mel_mask)
            attn_weights.append(w)
        return self.proj(out), attn_weights