# FastSpeech 2 on LibriTTS

Non-autoregressive TTS trained from scratch on the Quest HPC cluster (Northwestern).

## Pipeline

```
1. setup_env   →  conda env, MFA models, W&B login
2. download    →  LibriTTS subsets via parallel wget
3. align       →  MFA forced alignment (array job, 16 workers/subset)
4. preprocess  →  mel / pitch / energy / duration extraction (array job, 16 workers/subset)
5. train       →  DDP on 8× A100s via torchrun + SLURM
```

## Quick Start (on Quest)

```bash
# Clone and enter repo
cd /projects/e32706/speech_synthesis
git clone https://github.com/kanaghasivakumar/Speech-Synthesis && cd Speech-Synthesis

# 1. Environment
bash scripts/setup_env.sh

# 2. Download data (runs on login node, parallelised)
bash data/download_libriTTS.sh /projects/e32706/speech_synthesis/data/LibriTTS 8

# 3-5. SLURM jobs (submit in order, or chain with --dependency)
sbatch slurm/align.slurm
sbatch --dependency=afterok:<align_jid> slurm/preprocess.slurm
sbatch --dependency=afterok:<preprocess_jid> slurm/train.slurm
```

## Config

All hyperparameters and paths live in `configs/fastspeech2.yaml`. Update `paths.*` to match your allocation before running.

## Repo Layout

```
configs/          experiment config
data/             download + preprocessing shell scripts
slurm/            SLURM job scripts
src/
  model/          FastSpeech2, Encoder, Decoder, VarianceAdaptor
  modules/        FFTBlock, MultiHeadAttention, LengthRegulator, VariancePredictor
  data/           Dataset + collate
  utils/          audio feature extraction, text → phoneme → ID
  loss.py         multi-term MSE loss
  train.py        DDP training loop with AMP + W&B
  preprocess.py   parallel preprocessing worker
```

## Experiment Tracking

W&B logs: loss components, LR schedule, eval loss, and periodic audio samples.