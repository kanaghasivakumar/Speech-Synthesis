import os
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
from omegaconf import OmegaConf

from src.model.fastspeech2 import FastSpeech2
from src.utils.text import text_to_ids


def infer(text, model, cfg, device, p_control=1.0, e_control=1.0, d_control=1.0):
    ids = text_to_ids(text)
    phonemes = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    src_lens = torch.tensor([len(ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        mel_out, mel_post, *_ = model(
            phonemes, src_lens,
            p_control=p_control, e_control=e_control, d_control=d_control
        )

    mel = mel_post.squeeze(0).cpu().numpy()
    return mel


def mel_to_audio(mel, cfg):
    mel_db = mel.T
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=cfg.data.sampling_rate,
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        win_length=cfg.audio.win_length,
        fmin=cfg.audio.fmin,
        fmax=cfg.audio.fmax,
        n_iter=60,
    )
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fastspeech2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default="output.wav")
    parser.add_argument("--p_control", type=float, default=1.0)
    parser.add_argument("--e_control", type=float, default=1.0)
    parser.add_argument("--d_control", type=float, default=1.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastSpeech2(cfg).to(device)
    model.variance_adaptor.set_pitch_bins(cfg.audio.pitch_min, cfg.audio.pitch_max, cfg.audio.pitch_log_scale)
    model.variance_adaptor.set_energy_bins(cfg.audio.energy_min, cfg.audio.energy_max)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mel = infer(args.text, model, cfg, device, args.p_control, args.e_control, args.d_control)
    audio = mel_to_audio(mel, cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, audio, cfg.data.sampling_rate)
    print(f"saved {args.out} ({len(audio)/cfg.data.sampling_rate:.2f}s)")


if __name__ == "__main__":
    main()