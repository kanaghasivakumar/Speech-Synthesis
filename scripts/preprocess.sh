#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-slurm}"

if [[ "$MODE" == "slurm" ]]; then
    JID=$(sbatch --parsable slurm/preprocess.slurm)
    echo "Submitted preprocess array job: $JID"
    echo "Monitor: squeue -j $JID"
    echo "Logs: /projects/e32706/speech_synthesis/logs/preprocess_${JID}_*.out"

elif [[ "$MODE" == "local" ]]; then
    SUBSETS=("train-clean-100" "train-clean-360" "dev-clean" "test-clean")
    for SUBSET in "${SUBSETS[@]}"; do
        echo "Preprocessing $SUBSET..."
        python -m src.preprocess \
            --config configs/fastspeech2.yaml \
            --split "$SUBSET" \
            --workers "${2:-8}"
    done

else
    echo "Usage: $0 [slurm|local] [n_workers]"
    exit 1
fi