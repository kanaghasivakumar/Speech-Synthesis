# FastSpeech 2 on LibriTTS

Non-autoregressive TTS trained from scratch on Quest.

## Pipeline

```
1. setup       →  conda env (fastspeech2), MFA env (micromamba), W&B login
2. download    →  LibriTTS train-clean-100, train-clean-360, dev-clean, test-clean
3. align       →  Montreal Forced Aligner, english_us_arpa acoustic model
4. preprocess  →  mel spectrogram, pitch (RAPT F0), energy (RMS), duration extraction
5. train       →  DDP on 2× A100s via torchrun + SLURM, 322,800 steps
6. vocoder     →  HiFi-GAN fine-tuned on teacher-forced predicted mels
7. inference   →  text → phonemes → FastSpeech2 → mel → HiFi-GAN → wav
```

## Quick Start (on Quest)

```bash
git clone https://github.com/kanaghasivakumar/Speech-Synthesis
cd Speech-Synthesis
conda activate fastspeech2
```

MFA alignment (uses separate micromamba env):
```bash
export PATH="/projects/e32706/omb8654/conda/envs/mfa_env/bin:$PATH"
export MFA_ROOT_DIR=/projects/e32706/omb8654/speech_synthesis/mfa_temp
sbatch slurm/align.slurm
```

Preprocessing and training:
```bash
sbatch --dependency=afterok:<align_jid> slurm/preprocess.slurm
sbatch --dependency=afterok:<preprocess_jid> slurm/train.slurm
```

Training is self-resubmitting via SIGTERM trap. Resubmits automatically at the 2-hour SLURM limit and resumes from the latest checkpoint.

## Config

All hyperparameters and paths live in `configs/fastspeech2.yaml`. Update `paths.*` to match your allocation before running.

## Inference

```bash
python -m src.infer_gan \
    --checkpoint /path/to/step_322800.pt \
    --text "Your text here." \
    --out output.wav \
    --vocoder_checkpoint /path/to/hifigan/ft_checkpoints/g_00020000 \
```

## XAI Visualization

```bash
python -m src.visualize_xai \
    --checkpoint /path/to/step_322800.pt \
    --text "Your text here." \
    --out_dir xai_figures
```

Outputs encoder self-attention heatmaps for all 4 layers (mean and per-head) and a variance adaptor panel showing mel spectrogram, per-phoneme duration, pitch contour, and energy contour with phoneme boundary overlays.

## HiFi-GAN Fine-tuning

Teacher-forced mel extraction:
```bash
python -m src.extract_ft_mels \
    --checkpoint /path/to/step_322800.pt \
    --out_dir /path/to/ft_mels
```

Then submit the fine-tuning job:
```bash
./scripts/hifigan_ft.sh slurm
```

## Repo Layout

```
configs/          experiment config
slurm/            SLURM job scripts (align, preprocess, train, hifigan_ft)
scripts/          hifigan_ft.sh wrapper
src/
  model/          FastSpeech2, Encoder, Decoder, VarianceAdaptor
  modules/        FFTBlock, MultiHeadAttention, LengthRegulator, VariancePredictor
  data/           LibriTTSDataset, collate_fn
  utils/          audio feature extraction, text → phoneme → ID pipeline
  loss.py         six-term MSE loss with pitch/energy normalization
  train.py        DDP training loop with AMP, Noam LR, W&B logging
  preprocess.py   parallel preprocessing worker
  infer_gan.py    inference with HiFi-GAN vocoder
  extract_ft_mels.py  teacher-forced mel extraction for vocoder finetuning
  visualize_xai.py    encoder attention heatmaps + variance adaptor plots
```

## Results

Final losses at step 322,800: mel 0.553, postnet 0.549, duration 0.089, pitch 0.975, energy 0.0006, val loss 3.46.

W&B training logs: https://api.wandb.ai/links/kanaghasivakumar-northwestern-university/epeiplac

## Architecture

FastSpeech2 with 4-layer FFT encoder and decoder (hidden dim 256, 2 attention heads, kernel size 9), variance adaptor with log-scale pitch bins and 256 quantization bins, PostNet for mel refinement. Trained with mixed precision (AMP) and Noam learning rate schedule (warmup 4,000 steps). Phoneme inputs via g2p_en with a 86-token vocabulary. Features: 80-bin mel spectrogram at 24kHz, hop length 256, RAPT F0 pitch, RMS energy.