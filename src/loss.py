import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel_w = cfg.loss.mel_weight
        self.post_w = cfg.loss.postnet_weight
        self.dur_w = cfg.loss.duration_weight
        self.pitch_w = cfg.loss.pitch_weight
        self.energy_w = cfg.loss.energy_weight

    def forward(self, preds, targets, src_mask, mel_mask):
        mel_out, mel_post, log_dur_pred, pitch_pred, energy_pred = preds
        mel_target, duration_target, pitch_target, energy_target = targets

        src_mask_inv = ~src_mask
        mel_mask_inv = ~mel_mask

        T = mel_out.size(1)
        pitch_target = pitch_target[:, :T]
        energy_target = energy_target[:, :T]

        mel_loss = F.mse_loss(
            mel_out.masked_select(mel_mask_inv.unsqueeze(-1)),
            mel_target.masked_select(mel_mask_inv.unsqueeze(-1))
        )
        post_loss = F.mse_loss(
            mel_post.masked_select(mel_mask_inv.unsqueeze(-1)),
            mel_target.masked_select(mel_mask_inv.unsqueeze(-1))
        )
        dur_loss = F.mse_loss(
            log_dur_pred.masked_select(src_mask_inv),
            torch.log(duration_target.float().masked_select(src_mask_inv) + 1)
        )
        pitch_loss = F.mse_loss(
            pitch_pred.masked_select(mel_mask_inv),
            pitch_target.masked_select(mel_mask_inv)
        )
        energy_loss = F.mse_loss(
            energy_pred.masked_select(mel_mask_inv),
            energy_target.masked_select(mel_mask_inv)
        )

        total = (self.mel_w * mel_loss + self.post_w * post_loss +
                 self.dur_w * dur_loss + self.pitch_w * pitch_loss +
                 self.energy_w * energy_loss)

        return total, mel_loss, post_loss, dur_loss, pitch_loss, energy_loss