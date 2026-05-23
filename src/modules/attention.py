import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.w_q(q).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_k)
        return self.fc(out), attn