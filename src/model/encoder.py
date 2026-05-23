import math
import torch
import torch.nn as nn
from src.modules.fft_block import FFTBlock


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.text.phoneme_vocab_size, cfg.model.encoder_hidden, padding_idx=0)
        self.pos_enc = PositionalEncoding(cfg.model.encoder_hidden, cfg.model.encoder_dropout, cfg.model.max_seq_len)
        self.layers = nn.ModuleList([
            FFTBlock(
                cfg.model.encoder_hidden,
                cfg.model.encoder_head,
                cfg.model.fft_conv_filter_size,
                cfg.model.encoder_kernel_size,
                cfg.model.encoder_dropout,
            )
            for _ in range(cfg.model.encoder_layers)
        ])

    def forward(self, x, src_mask):
        out = self.pos_enc(self.embed(x))
        attn_weights = []
        for layer in self.layers:
            out, w = layer(out, src_mask)
            attn_weights.append(w)
        return out, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])