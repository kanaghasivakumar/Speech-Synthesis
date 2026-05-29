import numpy as np
import torch
import librosa
import pyworld as pw


def load_wav(path, sr):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float64)


def mel_spectrogram(wav, cfg):
    stft = librosa.stft(
        wav.astype(np.float32),
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        win_length=cfg.audio.win_length,
    )
    mag = np.abs(stft)
    mel_basis = librosa.filters.mel(
        sr=cfg.data.sampling_rate,
        n_fft=cfg.audio.n_fft,
        n_mels=cfg.audio.n_mels,
        fmin=cfg.audio.fmin,
        fmax=cfg.audio.fmax,
    )
    mel = np.dot(mel_basis, mag)
    mel = np.log(np.maximum(mel, 1e-5))
    return mel.T.astype(np.float32)


def extract_pitch(wav, sr, hop_length):
    f0, t = pw.dio(wav, sr, frame_period=hop_length / sr * 1000)
    f0 = pw.stonemask(wav, f0, t, sr)
    return f0.astype(np.float32)


def extract_energy(wav, n_fft, hop_length, win_length):
    stft = librosa.stft(
        wav.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    energy = np.sqrt(np.sum(np.abs(stft) ** 2, axis=0))
    return energy.astype(np.float32)


def normalize_pitch(f0, mean, std):
    mask = f0 > 0
    f0 = f0.copy()
    f0[mask] = (f0[mask] - mean) / std
    return f0


def to_mel_tensor(mel_np):
    return torch.from_numpy(mel_np)