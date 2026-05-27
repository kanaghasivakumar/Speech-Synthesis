#!/bin/bash
#SBATCH --job-name=fs2_align
#SBATCH --account=e32706
#SBATCH --partition=normal
#SBATCH --array=1-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/e32706/speech_synthesis/logs/align_%A_%a.out
#SBATCH --error=/projects/e32706/speech_synthesis/logs/align_%A_%a.err

module purge
module load gcc/12.3.0-gcc
module load cuda/12.1.0-gcc-11.2.0

export PATH="/projects/e32706/omb8654/conda/envs/mfa_env/bin:$PATH"

SUBSETS=("train-clean-100" "train-clean-360")
SUBSET="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"

ALIGN_OUT=/projects/e32706/speech_synthesis/data/alignments/${SUBSET}
CORPUS_DIR=/projects/e32706/speech_synthesis/data/mfa_corpus/${SUBSET}

mkdir -p "$ALIGN_OUT" /projects/e32706/speech_synthesis/logs

export MFA_ROOT_DIR=/projects/e32706/speech_synthesis/mfa_temp
mkdir -p $MFA_ROOT_DIR

/projects/e32706/omb8654/conda/envs/mfa_env/bin/mfa align \
    "$CORPUS_DIR" \
    english_us_arpa \
    english_us_arpa \
    "$ALIGN_OUT" \
    --num_jobs "$SLURM_CPUS_PER_TASK" \
    --beam 10 \
    --retry_beam 40 \
    --clean