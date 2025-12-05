#!/bin/bash
# SLURM job script for 3 nodes x 8 GPUs (24 processes) using accelerate
# Fill in partition/account and model paths before submission.

#SBATCH --job-name=grpo
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --exclusive

set -euo pipefail

# Load environment (adjust to your cluster)
module load cuda/12.1 || true
source /path/to/venv/bin/activate

# NCCL networking hints (tune to your interfaces)
export NCCL_DEBUG=warn
export NCCL_SOCKET_IFNAME=eth0          # change to ib0/enp* as needed
export NCCL_IB_DISABLE=0                # set to 1 if no InfiniBand
export OMP_NUM_THREADS=2

# Master address/port from the first node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

CONFIG=grpo/eight_gpu.yaml

srun --nodes=3 --ntasks=3 --ntasks-per-node=1 bash -c '
  RANK=$SLURM_PROCID
  accelerate launch \
    --config_file '"$CONFIG"' \
    --machine_rank $RANK \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    grpo/grpo_gpu.py \
      --model_name_or_path <model_path> \
      --output_dir <output_dir> \
      --learning_rate 1e-5 \
      --gradient_checkpointing \
      --dtype bfloat16 \
      --max_prompt_length 2048 \
      --max_completion_length 1024 \
      --use_peft \
      --lora_target_modules "q_proj" "v_proj" \
      --reward_model_path <reward_model_path>
'
