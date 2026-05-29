#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-slurm}"
RESUME="${2:-}"

if [[ "$MODE" == "slurm" ]]; then
    if [[ -n "$RESUME" ]]; then
        JID=$(sbatch --parsable --export=ALL,RESUME_CKPT="$RESUME" slurm/train.slurm)
    else
        JID=$(sbatch --parsable slurm/train.slurm)
    fi
    echo "Submitted training job: $JID"
    echo "Monitor: squeue -j $JID"
    echo "Logs: /projects/e32706/omb8654/speech_synthesis/logs/train_${JID}.out"
    echo "W&B:  https://wandb.ai"

elif [[ "$MODE" == "interactive" ]]; then
    LATEST=$(ls -t /projects/e32706/omb8654/speech_synthesis/checkpoints/step_*.pt 2>/dev/null | head -n1 || true)
    RESUME_FLAG=""
    [[ -n "$LATEST" ]] && RESUME_FLAG="--resume $LATEST"
    [[ -n "$RESUME" ]] && RESUME_FLAG="--resume $RESUME"

    torchrun \
        --nnodes=1 \
        --nproc_per_node="$(python -c 'import torch; print(torch.cuda.device_count())')" \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        -m src.train \
        --config configs/fastspeech2.yaml \
        $RESUME_FLAG

else
    echo "Usage: $0 [slurm|interactive] [path/to/checkpoint.pt]"
    exit 1
fi