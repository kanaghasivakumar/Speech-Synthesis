#!/bin/bash
#SBATCH --job-name=fs2_train
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/e32706/omb8654/speech_synthesis/logs/train_%j.out
#SBATCH --error=/projects/e32706/omb8654/speech_synthesis/logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kanaghasivakumar2026@u.northwestern.edu

module purge
module load gcc/12.3.0-gcc
module load cuda/12.1.0-gcc-11.2.0

source /home/omb8654/miniconda3/etc/profile.d/conda.sh
conda activate fastspeech2

export PYTHONPATH=/projects/e32706/omb8654/Speech-Synthesis
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(shuf -i 29500-29999 -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export WANDB_DIR=/projects/e32706/omb8654/speech_synthesis/wandb

RESUME=""
LATEST=$(ls -t /projects/e32706/omb8654/speech_synthesis/checkpoints/step_*.pt 2>/dev/null | head -n1)
[[ -n "$LATEST" ]] && RESUME="--resume $LATEST"

cd /projects/e32706/omb8654/Speech-Synthesis

mkdir -p /projects/e32706/omb8654/speech_synthesis/wandb/wandb
chmod -R 775 /projects/e32706/omb8654/speech_synthesis/wandb

resubmit() {
    LATEST=$(ls -t /projects/e32706/omb8654/speech_synthesis/checkpoints/step_*.pt 2>/dev/null | head -n1)
    [[ -n "$LATEST" ]] && sbatch --account=p32576 /projects/e32706/omb8654/Speech-Synthesis/slurm/train.slurm
}
trap resubmit SIGTERM

torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=2 \
    --master_port="$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m src.train \
    --config configs/fastspeech2.yaml \
    $RESUME &
wait $!