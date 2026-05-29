#!/usr/bin/env bash
set -euo pipefail

module load gcc/12.3.0-gcc
module load cuda/12.1.0-gcc-11.2.0

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
	source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

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

    export CUDA_NVML_BARE_METAL=1
    torchrun \
        --nnodes=1 \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        -m src.train \
        --config configs/fastspeech2.yaml \
        $RESUME_FLAG

else
    echo "Usage: $0 [slurm|interactive] [path/to/checkpoint.pt]"
    exit 1
fi
