#!/bin/bash
#SBATCH --job-name=fs2_preprocess
#SBATCH --account=e32706
#SBATCH --partition=normal
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/e32706/speech_synthesis/logs/preprocess_%A_%a.out
#SBATCH --error=/projects/e32706/speech_synthesis/logs/preprocess_%A_%a.err

module purge
module load gcc/12.3.0

source ~/.bashrc
conda activate fastspeech2

SUBSETS=("train-clean-100" "train-clean-360")
SUBSET="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"

python -m src.preprocess \
    --config configs/fastspeech2.yaml \
    --split "$SUBSET" \
    --workers "$SLURM_CPUS_PER_TASK"