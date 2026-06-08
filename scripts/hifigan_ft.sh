#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-slurm}"

if [[ "$MODE" == "slurm" ]]; then
    JID=$(sbatch --parsable slurm/hifigan_ft.slurm)
    echo "Submitted HiFi-GAN finetuning job: $JID"
    echo "Monitor: squeue -j $JID"
    echo "Logs: /projects/e32706/omb8654/speech_synthesis/logs/hifigan_ft_${JID}.out"

elif [[ "$MODE" == "local" ]]; then
    echo "Starting local HiFi-GAN finetuning..."
    cd /projects/e32706/omb8654/hifi-gan

    python train.py \
        --config config_v1.json \
        --fine_tuning True \
        --checkpoint_path ft_checkpoints \
        --input_wavs_dir /projects/e32706/omb8654/speech_synthesis/data/LibriTTS/train-clean-100 \
        --input_mels_dir /projects/e32706/omb8654/speech_synthesis/data/ft_mels \
        --training_epochs 5000 \
        --input_training_file /projects/e32706/omb8654/Speech-Synthesis/hifigan_train.txt \
        --input_validation_file /projects/e32706/omb8654/Speech-Synthesis/hifigan_val.txt
else
    echo "Usage: $0 [slurm|local]"
    exit 1
fi
