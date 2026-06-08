#!/bin/bash
#SBATCH --job-name=hifi_ft
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/e32706/omb8654/speech_synthesis/logs/hifigan_ft_%j.out
#SBATCH --error=/projects/e32706/omb8654/speech_synthesis/logs/hifigan_ft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kanaghasivakumar2026@u.northwestern.edu

module purge
module load gcc/12.3.0-gcc
module load cuda/12.1.0-gcc-11.2.0

source /home/omb8654/miniconda3/etc/profile.d/conda.sh
conda activate fastspeech2

# Execute the fine-tuning from the hifi-gan directory
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
