#!/bin/bash
#SBATCH --job-name=reward_modeling_accelerate
#SBATCH --output=reward_modeling/logs/%j.out
#SBATCH --error=reward_modeling/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --chdir=/data/scratch/irisxu/classes/deep_learning/dl-rm-loss
# Activate conda environment (assumes conda init already in ~/.bashrc)
source ~/.bashrc
conda activate occ-llm

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p reward_modeling/logs

# Accelerate config file (4 GPU config)
ACCELERATE_CONFIG="examples/accelerate_configs/four_gpu.yaml"

OUTPUT_DIR="reward_modeling/Qwen2-0.5B-Reward-GCE"
MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME="datasets/ultrafeedback_binarized_clean"

accelerate launch --config_file "$ACCELERATE_CONFIG" \
    reward_modeling/reward_modeling.py \
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

