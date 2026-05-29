#!/bin/bash
#SBATCH --job-name=fs2_train
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/e32706/omb8654/speech_synthesis/logs/train_%j.out
#SBATCH --error=/projects/e32706/omb8654/speech_synthesis/logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kanaghasivakumar2026@u.northwestern.edu

module purge
module load cuda/12.1.1
module load gcc/12.3.0

source ~/.bashrc
conda activate fastspeech2

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export WANDB_DIR=/projects/e32706/omb8654/speech_synthesis/wandb

RESUME=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LATEST=$(ls -t /projects/e32706/omb8654/speech_synthesis/checkpoints/step_*.pt 2>/dev/null | head -n1)
    [[ -n "$LATEST" ]] && RESUME="--resume $LATEST"
fi

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m src.train \
    --config configs/fastspeech2.yaml \
    $RESUME