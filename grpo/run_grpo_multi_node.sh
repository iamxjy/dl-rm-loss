#!/bin/bash
# LSF job script for 3 nodes x 8 GPUs (24 processes) using accelerate
# Fill in queue/project paths before submission.

#BSUB -J grpo
#BSUB -q normal
#BSUB -G grp_runtime
#BSUB -n 3
#BSUB -R "span[ptile=1]"
#BSUB -W 24:00
#BSUB -M 5T
#BSUB -gpu "num=8/task:j_exclusive=yes:mode=shared"
#BSUB -env "all"
#BSUB -o grpo/logs/%J.out
#BSUB -e grpo/logs/%J.err

set -euo pipefail

# Explicitly redirect all output to log files (in case blaunch doesn't capture it)
exec > >(tee -a grpo/logs/${LSB_JOBID:-unknown}.out)
exec 2> >(tee -a grpo/logs/${LSB_JOBID:-unknown}.err >&2)

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Load environment (adjust to your cluster)
cd /proj/inf-scaling/iris/dl-rm-loss
echo "=== GRPO Multi-Node Job Started ===" >&2
echo "Hostname: $(hostname)" >&2
echo "LSB_HOSTS: $LSB_HOSTS" >&2
echo "Working directory: $(pwd)" >&2
echo "Date: $(date)" >&2

source venv/bin/activate
echo "Virtual environment activated" >&2

# Create logs directory if it doesn't exist
mkdir -p grpo/logs

# Set HuggingFace cache directories to project space (avoid disk quota issues)
export HF_HOME="/proj/inf-scaling/iris/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="/proj/inf-scaling/iris/datasets_cache"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
echo "HuggingFace cache directories set to project space" >&2

# NCCL networking hints (tune to your interfaces)
export NCCL_DEBUG=INFO  # Changed from warn to INFO for better debugging
# export NCCL_SOCKET_IFNAME=ib0          # change to ib0/enp* as needed
export NCCL_IB_DISABLE=0                # set to 1 if no InfiniBand
export OMP_NUM_THREADS=2

# Parse LSB_HOSTS to get node list
NODE_LIST=($LSB_HOSTS)
MASTER_NODE=$(echo ${NODE_LIST[0]} | cut -d'.' -f1)
MASTER_PORT=29500

echo "Master node: $MASTER_NODE" >&2
echo "Node list: ${NODE_LIST[@]}" >&2

# Pre-download models on main node before distributed launch
# This prevents all ranks from trying to download simultaneously
if [[ $(hostname | cut -d'.' -f1) == "$MASTER_NODE" ]]; then
    echo "=== Pre-downloading models on master node ===" >&2
    bash grpo/download_model.sh
    echo "=== Model download complete ===" >&2
fi

# Wait for all nodes to be ready
sleep 5

# Generate launch script for each node
LAUNCH_SCRIPT="grpo/logs/launch_on_node_${LSB_JOBID}.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCH_EOF'
#!/bin/bash
# Determine rank from position in LSB_HOSTS
RANK=0
CURRENT_HOST=$(hostname | cut -d'.' -f1)
idx=0
for host in $LSB_HOSTS; do
  if [ "$host" = "$CURRENT_HOST" ]; then
    RANK=$idx
    break
  fi
  idx=$((idx + 1))
done

echo "=== Node $CURRENT_HOST starting (Rank $RANK) ===" >&2

cd /proj/inf-scaling/iris/dl-rm-loss
source venv/bin/activate

accelerate launch \
  --config_file grpo/eight_gpu.yaml \
  --machine_rank $RANK \
  --main_process_ip MASTER_NODE_PLACEHOLDER \
  --main_process_port 29500 \
  grpo/grpo_gpu.py \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --output_dir grpo/grpo-Qwen3-4B-Base \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --fp16 \
    --dtype float16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --use_peft \
    --lora_target_modules "q_proj" "v_proj" \
    --reward_model_path reward_modeling/models_final/Qwen3-0.6B-Base-noisy40-BT/checkpoint-969/ \
    --report_to wandb \
    --logging_steps 10 \
    --run_name grpo-Qwen3-4B-$(date +%Y%m%d-%H%M%S) 2>&1
LAUNCH_EOF

# Replace placeholder with actual master node
sed -i "s/MASTER_NODE_PLACEHOLDER/$MASTER_NODE/g" "$LAUNCH_SCRIPT"
chmod +x "$LAUNCH_SCRIPT"

echo "=== Launch script created: $LAUNCH_SCRIPT ===" >&2
cat "$LAUNCH_SCRIPT" >&2

# Launch training on all nodes using blaunch
echo "=== Launching distributed training with blaunch ===" >&2
echo "Launching on all allocated nodes..." >&2

# blaunch without -z will use all allocated nodes in the current job
# The script will run on each node
blaunch bash "$LAUNCH_SCRIPT"

BLAUNCH_EXIT=$?
echo "=== blaunch completed with exit code: $BLAUNCH_EXIT ===" >&2

if [ $BLAUNCH_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $BLAUNCH_EXIT" >&2
    exit $BLAUNCH_EXIT
fi

echo "=== GRPO Multi-Node Job Completed ===" >&2
