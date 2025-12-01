#!/bin/bash
#SBATCH --job-name=reward_modeling
#SBATCH --output=reward_modeling/logs/%j.out
#SBATCH --error=reward_modeling/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Script to run reward modeling training
# Loss type options: bradley_terry or gce

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get project root (parent directory of reward_modeling/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Change to project root directory
cd "$PROJECT_ROOT"

# Initialize conda (if not already initialized)
# Uncomment if needed:
# source ~/.bashrc
# or
# source /path/to/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
# Update this with your conda environment name
conda activate your_env_name

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/reward_modeling/logs"

OUTPUT_DIR="reward_modeling/Qwen2-0.5B-Reward-GCE"
MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME="datasets/ultrafeedback_binarized_clean"

python reward_modeling/reward_modeling.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --loss_type gce \
    --gce_q 0.7

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "End Time: $(date)"
else
    echo "Training failed with exit code $?"
    echo "End Time: $(date)"
    exit 1
fi

