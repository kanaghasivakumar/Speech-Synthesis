import os
import sys
import json
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
from omegaconf import OmegaConf

sys.path.insert(0, '/projects/e32706/omb8654/hifi-gan')
from models import Generator
from env import AttrDict

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


def load_hifigan(device):
    with open('/projects/e32706/omb8654/hifi-gan/config_v1.json') as f:
        h = AttrDict(json.load(f))
    generator = Generator(h).to(device)
    ckpt = torch.load('/projects/e32706/omb8654/hifi-gan/generator_universal.pth.tar', map_location=device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator, h


def mel_to_audio(mel, cfg, generator, h):
    device = next(generator.parameters()).device
    mel_tensor = torch.tensor(mel).T.unsqueeze(0).to(device)
    with torch.no_grad():
        audio = generator(mel_tensor).squeeze().cpu().numpy()
    audio = librosa.resample(audio, orig_sr=h.sampling_rate, target_sr=cfg.data.sampling_rate)
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
    normalized_bins = (model.variance_adaptor.pitch_bins - cfg.audio.pitch_mean) / cfg.audio.pitch_std
    model.variance_adaptor.pitch_bins.copy_(normalized_bins)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    generator, h = load_hifigan(device)

    mel = infer(args.text, model, cfg, device, args.p_control, args.e_control, args.d_control)
    audio = mel_to_audio(mel, cfg, generator, h)
    audio = audio / (np.abs(audio).max() + 1e-8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, (audio * 32767).astype(np.int16), cfg.data.sampling_rate, subtype='PCM_16')
    print(f"saved {args.out} ({len(audio)/cfg.data.sampling_rate:.2f}s)")


if __name__ == "__main__":
    main()