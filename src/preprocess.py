import os
import argparse
import json
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.audio import load_wav, mel_spectrogram, extract_pitch, extract_energy
from src.utils.text import phonemes_to_ids


def parse_mfa_textgrid(tg_path):
    from textgrid import TextGrid
    tg = TextGrid.fromFile(tg_path)
    phoneme_tier = None
    for tier in tg.tiers:
        if tier.name.lower() in ("phones", "phone"):
            phoneme_tier = tier
            break
    if phoneme_tier is None:
        return None
    return [(iv.minTime, iv.maxTime, iv.mark) for iv in phoneme_tier]


def process_one(uid, wav_path, tg_path, text, out_dir, cfg):
    out_base = os.path.join(out_dir, uid)
    if os.path.exists(f"{out_base}_mel.npy"):
        return uid, True

    intervals = parse_mfa_textgrid(tg_path)
    if intervals is None:
        return uid, False

    wav = load_wav(wav_path, cfg.data.sampling_rate)
    mel = mel_spectrogram(wav, cfg)
    pitch = extract_pitch(wav, cfg.data.sampling_rate, cfg.audio.hop_length)
    energy = extract_energy(wav, cfg.audio.n_fft, cfg.audio.hop_length, cfg.audio.win_length)

    T_mel = mel.shape[0]
    pitch = pitch[:T_mel]
    energy = energy[:T_mel]

    phonemes, durations = [], []
    for start, end, ph in intervals:
        if ph in ("", "sp", "sil", "<eps>"):
            ph = "spn"
        frames = max(1, round((end - start) * cfg.data.sampling_rate / cfg.audio.hop_length))
        phonemes.append(ph)
        durations.append(frames)

    total_dur = sum(durations)
    if abs(total_dur - T_mel) > 3:
        scale = T_mel / total_dur
        durations = [max(1, round(d * scale)) for d in durations]
        durations[-1] += T_mel - sum(durations)

    ph_ids = phonemes_to_ids(phonemes)
    np.save(f"{out_base}_phonemes.npy", np.array(ph_ids, dtype=np.int32))
    np.save(f"{out_base}_mel.npy", mel)
    np.save(f"{out_base}_pitch.npy", pitch)
    np.save(f"{out_base}_energy.npy", energy)
    np.save(f"{out_base}_duration.npy", np.array(durations, dtype=np.int32))

    return uid, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fastspeech2.yaml")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = os.path.join(cfg.paths.preprocessed, args.split)
    os.makedirs(out_dir, exist_ok=True)

    libri_root = os.path.join(cfg.paths.libriTTS_root, args.split)
    align_root = os.path.join(cfg.paths.mfa_alignments, args.split)

    wav_files = sorted(glob.glob(f"{libri_root}/**/*.wav", recursive=True))

    jobs = []
    for wav_path in wav_files:
        p = Path(wav_path)
        uid = p.stem
        spk, chap = p.parts[-3], p.parts[-2]
        tg_path = os.path.join(align_root, f"{uid}.TextGrid")
        if not os.path.exists(tg_path):
            tg_path = os.path.join(align_root, spk, f"{uid}.TextGrid")
        if not os.path.exists(tg_path):
            tg_path = os.path.join(align_root, spk, chap, f"{uid}.TextGrid")
        txt_path = wav_path.replace(".wav", ".normalized.txt")
        if not os.path.exists(tg_path) or not os.path.exists(txt_path):
            continue
        with open(txt_path) as f:
            text = f.read().strip()
        jobs.append((uid, wav_path, tg_path, text, out_dir, cfg))

    success, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_one, *j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=args.split):
            uid, ok = fut.result()
            if ok:
                success += 1
            else:
                failed += 1

    manifest = os.path.join(cfg.paths.preprocessed, f"{args.split}_manifest.txt")
    with open(manifest, "w") as f:
        processed = [Path(p).stem for p in glob.glob(f"{out_dir}/*_mel.npy")]
        f.write("\n".join(sorted(processed)))

    print(f"Done: {success} success, {failed} failed -> {manifest}")


if __name__ == "__main__":
    main()