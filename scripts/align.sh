#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-slurm}"

if [[ "$MODE" == "slurm" ]]; then
    JID=$(sbatch --parsable slurm/align.slurm)
    echo "Submitted align array job: $JID"
    echo "Monitor: squeue -j $JID"
    echo "Logs: /projects/e32706/speech_synthesis/logs/align_${JID}_*.out"

elif [[ "$MODE" == "local" ]]; then
    CONFIG="${2:-configs/fastspeech2.yaml}"
    SUBSET="${3:-train-clean-100}"
    WORKERS="${4:-8}"

    LIBRI_ROOT=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); print(c.paths.libriTTS_root)")/${SUBSET}
    ALIGN_OUT=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); print(c.paths.mfa_alignments)")/${SUBSET}
    CORPUS_DIR=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); print(c.paths.data_root)")/mfa_corpus/${SUBSET}

    mkdir -p "$CORPUS_DIR" "$ALIGN_OUT"

    find "$LIBRI_ROOT" -name "*.wav" | parallel -j "$WORKERS" \
        'WAV={}; BASE="${WAV%.wav}"; \
        ln -sf "$WAV" '"$CORPUS_DIR"'/"$(basename $WAV)"; \
        cp "${BASE}.normalized.txt" '"$CORPUS_DIR"'/"$(basename ${BASE}).txt" 2>/dev/null || true'

    mfa align \
        --num_jobs "$WORKERS" \
        --beam 10 \
        --retry_beam 40 \
        --clean \
        "$CORPUS_DIR" \
        english_us_arpa \
        english_us_arpa \
        "$ALIGN_OUT"
else
    echo "Usage: $0 [slurm|local] [config] [subset] [n_workers]"
    exit 1
fi