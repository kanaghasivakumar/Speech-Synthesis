import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.fastspeech2 import FastSpeech2
from src.utils.text import text_to_phonemes, phonemes_to_ids, PHONEME_VOCAB


def load_model(cfg, ckpt_path, device):
    model = FastSpeech2(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.variance_adaptor.set_pitch_bins(
        cfg.audio.pitch_min, cfg.audio.pitch_max,
        cfg.audio.get("pitch_log_scale", True),
    )
    model.variance_adaptor.set_energy_bins(cfg.audio.energy_min, cfg.audio.energy_max)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, phoneme_ids, device):
    ph = torch.tensor(phoneme_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([ph.size(1)], device=device)

    mel_out, mel_post, log_dur_pred, pitch_pred, energy_pred, \
        src_mask, mel_mask, enc_attn, dec_attn = model(
            ph, src_lens, mel_lens=None, max_mel_len=None,
        )

    return {
        "phoneme_ids": phoneme_ids,
        "mel": mel_post.squeeze(0).cpu().numpy(),
        "dur": torch.clamp(torch.round(torch.exp(log_dur_pred) - 1), min=0)
               .squeeze(0).cpu().numpy(),
        "pitch": pitch_pred.squeeze(0).cpu().numpy(),
        "energy": energy_pred.squeeze(0).cpu().numpy(),
        "enc_attn": [w.squeeze(0).cpu().numpy() for w in enc_attn],
    }


def ids_to_labels(phoneme_ids):
    return [PHONEME_VOCAB[i] if i < len(PHONEME_VOCAB) else str(i) for i in phoneme_ids]


def plot_attention_heatmaps(results, out_dir, layer_indices=None):
    enc_attn = results["enc_attn"]
    labels = ids_to_labels(results["phoneme_ids"])
    n_layers = len(enc_attn)
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    os.makedirs(out_dir, exist_ok=True)

    for layer_idx in layer_indices:
        attn = enc_attn[layer_idx]
        n_head = attn.shape[0]
        mean_attn = attn.mean(axis=0)

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4 + 1),
                                        max(5, len(labels) * 0.4 + 1)))
        im = ax.imshow(mean_attn, aspect="auto", cmap="viridis", origin="upper",
                       vmin=0, vmax=mean_attn.max())
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.set_title(f"Encoder Layer {layer_idx + 1} Attention (mean, {n_head} heads)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(out_dir, f"enc_attn_layer{layer_idx+1}_mean.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(path)

        cols = min(n_head, 4)
        rows = (n_head + cols - 1) // cols
        cell = max(3, len(labels) * 0.25)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cell * cols + 1, cell * rows + 0.5),
                                 squeeze=False)
        for h in range(n_head):
            r, c = divmod(h, cols)
            ax = axes[r][c]
            ax.imshow(attn[h], aspect="auto", cmap="viridis", origin="upper",
                      vmin=0, vmax=attn[h].max())
            ax.set_title(f"Head {h + 1}", fontsize=8)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        for h in range(n_head, rows * cols):
            r, c = divmod(h, cols)
            axes[r][c].axis("off")
        fig.suptitle(f"Encoder Layer {layer_idx + 1} Attention (per head)", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f"enc_attn_layer{layer_idx+1}_per_head.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(path)


def plot_variance_adaptor(results, out_dir):
    mel = results["mel"]
    dur = results["dur"]
    pitch = results["pitch"]
    energy = results["energy"]
    labels = ids_to_labels(results["phoneme_ids"])
    T_mel = mel.shape[0]
    T_src = len(labels)

    os.makedirs(out_dir, exist_ok=True)

    cum = np.concatenate([[0], np.cumsum(dur)])
    phoneme_centers = (cum[:-1] + cum[1:]) / 2.0
    x = np.arange(T_mel)

    fig, axes = plt.subplots(4, 1, figsize=(max(10, T_mel * 0.04), 12),
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    ax = axes[0]
    img = ax.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_ylabel("Mel bin")
    ax.set_title("Mel spectrogram")
    for boundary in cum[1:-1]:
        ax.axvline(x=boundary, color="cyan", linewidth=0.5, alpha=0.6)
    fig.colorbar(img, ax=ax, fraction=0.02, pad=0.01)

    ax = axes[1]
    ax.bar(range(T_src), dur, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(T_src))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Frames")
    ax.set_title("Duration")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[2]
    ax.plot(x, pitch, color="darkorange", linewidth=1.0)
    ax.set_ylabel("Pitch")
    ax.set_title("Pitch contour")
    ax.set_xlim(0, T_mel - 1)
    for boundary in cum[1:-1]:
        ax.axvline(x=boundary, color="gray", linewidth=0.4, alpha=0.5)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(phoneme_centers[:T_src])
    ax2.set_xticklabels(labels, fontsize=6, rotation=45, ha="left")
    ax2.tick_params(length=0)

    ax = axes[3]
    ax.plot(x, energy, color="mediumseagreen", linewidth=1.0)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Frame")
    ax.set_title("Energy contour")
    ax.set_xlim(0, T_mel - 1)
    for boundary in cum[1:-1]:
        ax.axvline(x=boundary, color="gray", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "variance_adaptor_panel.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--config", default="configs/fastspeech2.yaml")
    parser.add_argument("--out_dir", default="xai_figures")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    model = load_model(cfg, args.checkpoint, device)

    phonemes = text_to_phonemes(args.text)
    ph_ids = phonemes_to_ids(phonemes)
    print(f"phonemes: {phonemes}")

    results = run_inference(model, ph_ids, device)

    n_layers = len(results["enc_attn"])
    layer_indices = args.layers
    if layer_indices is not None:
        layer_indices = [i for i in layer_indices if 0 <= i < n_layers]

    plot_attention_heatmaps(results, args.out_dir, layer_indices)
    plot_variance_adaptor(results, args.out_dir)


if __name__ == "__main__":
    main()
