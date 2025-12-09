#!/bin/bash

# Change to project root directory
cd /proj/inf-scaling/iris/dl-rm-loss

# Activate conda environment
source venv/bin/activate

# Noise variant then GCE q
NOISE="${1:-clean}"
GCE_Q="${2:-0.7}"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Noise: ${NOISE}"
echo "GCE_q: ${GCE_Q}"
# Create logs directory if it doesn't exist
mkdir -p reward_modeling/logs

python reward_modeling/reward_modeling.py \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --dataset_name datasets/ultrafeedback_binarized_$NOISE \
    --hf_cache_dir /proj/inf-scaling/iris/hf_cache \
    --output_dir reward_modeling/models_final/Qwen3-0.6B-Base-$NOISE-GCE-$GCE_Q \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --loss_type gce \
    --gce_q $GCE_Q

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "End Time: $(date)"
else
    echo "Training failed with exit code $?"
    echo "End Time: $(date)"
    exit 1
fi

