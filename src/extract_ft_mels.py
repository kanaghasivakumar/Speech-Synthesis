import os
import argparse
import glob
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path

# Modified import to work from inside the 'src' directory
from model.fastspeech2 import FastSpeech2

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    # Pointed the default config path up one directory since we are in src/
    parser.add_argument("--config", default="../configs/fastspeech2.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to your latest step_242000.pt")
    parser.add_argument("--out_dir", default="/projects/e32706/omb8654/speech_synthesis/data/ft_mels")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Model
    model = FastSpeech2(cfg).to(device)
    model.variance_adaptor.set_pitch_bins(cfg.audio.pitch_min, cfg.audio.pitch_max, cfg.audio.pitch_log_scale)
    model.variance_adaptor.set_energy_bins(cfg.audio.energy_min, cfg.audio.energy_max)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Strip 'module.' if it was saved as DDP
    state_dict = ckpt["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 2. Iterate through train-clean-100 preprocessed files
    prep_dir = os.path.join(cfg.paths.preprocessed, "train-clean-100")
    mel_files = glob.glob(f"{prep_dir}/*_mel.npy")
    
    print(f"Extracting predicted mels for {len(mel_files)} files...")
    
    for mel_path in tqdm(mel_files):
        uid = Path(mel_path).stem.replace("_mel", "")
        
        # Load ground truth features
        phonemes = torch.tensor(np.load(f"{prep_dir}/{uid}_phonemes.npy")).unsqueeze(0).to(device)
        pitch = torch.tensor(np.load(f"{prep_dir}/{uid}_pitch.npy")).unsqueeze(0).to(device)
        energy = torch.tensor(np.load(f"{prep_dir}/{uid}_energy.npy")).unsqueeze(0).to(device)
        duration = torch.tensor(np.load(f"{prep_dir}/{uid}_duration.npy")).unsqueeze(0).to(device)
        
        src_lens = torch.tensor([phonemes.size(1)]).to(device)
        mel_lens = torch.tensor([duration.sum().item()]).to(device)

        # 3. Teacher-forced forward pass (pass in ground truth pitch, energy, duration)
        _, mel_post, _, _, _, _, _, _, _ = model(
            phonemes, src_lens, mel_lens=mel_lens, max_mel_len=mel_lens[0],
            pitch=pitch, energy=energy, duration=duration
        )

        # 4. Save the predicted postnet mel 
        pred_mel = mel_post.squeeze(0).cpu().numpy() # Shape: (T, 80)
        
        # HiFi-GAN expects shape (80, T), so we transpose it
        pred_mel = pred_mel.T 
        np.save(os.path.join(args.out_dir, f"{uid}.npy"), pred_mel)

if __name__ == "__main__":
    main()