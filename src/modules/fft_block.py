import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, kernel_size, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, kernel_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x, attn_weights = self.attn(x, x, x, mask)
        x = self.norm1(residual + self.dropout(x))
        residual = x
        x = self.ff(x)
        x = self.norm2(residual + self.dropout(x))
        return x, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.act(self.conv1(x)))
        x = self.conv2(x)
        return x.transpose(1, 2)