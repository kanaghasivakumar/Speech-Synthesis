#!/bin/bash
#SBATCH --job-name=fs2_align
#SBATCH --account=p32349
#SBATCH --partition=normal
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/p32349/speech_synthesis/logs/align_%A_%a.out
#SBATCH --error=/projects/p32349/speech_synthesis/logs/align_%A_%a.err

module purge
module load gcc/12.3.0

source ~/.bashrc
conda activate fastspeech2

SUBSETS=("train-clean-100" "train-clean-360")
SUBSET="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"

LIBRI_ROOT=/projects/p32349/speech_synthesis/data/LibriTTS/${SUBSET}
ALIGN_OUT=/projects/p32349/speech_synthesis/data/alignments/${SUBSET}
CORPUS_DIR=/projects/p32349/speech_synthesis/data/mfa_corpus/${SUBSET}

mkdir -p "$CORPUS_DIR" "$ALIGN_OUT"

find "$LIBRI_ROOT" -name "*.wav" | parallel -j "$SLURM_CPUS_PER_TASK" \
    'WAV={}; BASE="${WAV%.wav}"; SPKCHAP=$(dirname "$WAV" | xargs dirname | xargs basename)_$(dirname "$WAV" | xargs basename); \
    mkdir -p '"$CORPUS_DIR"'; \
    ln -sf "$WAV" '"$CORPUS_DIR"'/"$(basename ${WAV%.wav})".wav; \
    cp "${BASE}.normalized.txt" '"$CORPUS_DIR"'/"$(basename ${BASE})".txt 2>/dev/null || true'

mfa align \
    --num_jobs "$SLURM_CPUS_PER_TASK" \
    --beam 10 \
    --retry_beam 40 \
    --clean \
    "$CORPUS_DIR" \
    english_us_arpa \
    english_us_arpa \
    "$ALIGN_OUT"