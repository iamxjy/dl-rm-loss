# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "trackio",
#     "kernels",
# ]
# ///

"""
python reward_modeling/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name datasets/ultrafeedback_binarized_clean \
    --output_dir Qwen2-0.5B-Reward-GCE \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --loss_type bradley_terry
"""

import os
from dataclasses import dataclass, field

# Disable trackio to avoid HuggingFace authentication issues
# (trackio requires HF token which causes errors on shared accounts)
os.environ.pop("TRACKIO_SPACE_ID", None)  # Remove if set

# Configure wandb for personal account FIRST (before ANY imports that might initialize wandb)
# Personal wandb config directory (keeps settings separate on shared accounts)
WANDB_PERSONAL_DIR = "/proj/inf-scaling/iris/iris_wandb_config"
WANDB_API_KEY_FILE = os.path.join(WANDB_PERSONAL_DIR, "api_key.txt")

# Try to load API key from file (created by setup_wandb.sh)
wandb_api_key = None
if os.path.exists(WANDB_API_KEY_FILE):
    with open(WANDB_API_KEY_FILE, 'r') as f:
        wandb_api_key = f.read().strip()

# Also check if API key is set as environment variable (takes precedence)
if os.getenv("WANDB_API_KEY"):
    wandb_api_key = os.getenv("WANDB_API_KEY")

# Set up wandb directories (MUST be set before any wandb imports)
os.environ["WANDB_CONFIG_DIR"] = WANDB_PERSONAL_DIR
os.environ["WANDB_CACHE_DIR"] = os.path.join(WANDB_PERSONAL_DIR, "cache")
os.environ["WANDB_DATA_DIR"] = os.path.join(WANDB_PERSONAL_DIR, "data")

# Create wandb directories if they don't exist
os.makedirs(WANDB_PERSONAL_DIR, exist_ok=True)
os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["WANDB_DATA_DIR"], exist_ok=True)

# Set API key if provided
if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key

# If HF_DATASETS_CACHE is set in environment, use it early (before imports)
# This will be overridden by command-line argument if provided
if "HF_DATASETS_CACHE" in os.environ:
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, HfArgumentParser

from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.reward_config_modified import RewardConfig
from trl.trainer.reward_trainer_modified import RewardTrainer


@dataclass
class RewardModelingScriptArguments(ScriptArguments):
    """
    Extended script arguments for reward modeling with cache directory support.
    """
    hf_cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "HuggingFace datasets cache directory. If not provided, uses HF_DATASETS_CACHE env var, "
            "or defaults to system default cache location."
        },
    )


logger = logging.get_logger(__name__)

# Print wandb status after imports (so we can use print safely)
if wandb_api_key:
    print(f"✓ Using personal wandb API key (config: {WANDB_PERSONAL_DIR})")
    print("✓ wandb logging enabled")
else:
    print("WARNING: WANDB_API_KEY not found. Disabling wandb for this run.")
    print("To enable wandb:")
    print(f"  1. Run: ./grpo/setup_wandb.sh (saves key to {WANDB_API_KEY_FILE})")
    print("  2. Or set environment variable: export WANDB_API_KEY='your_key_here'")
    print("  3. The API key file is gitignored and will NOT be committed to git")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":
    parser = HfArgumentParser((RewardModelingScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    # Ensure trackio is not in report_to (to avoid HF authentication errors)
    # If report_to is None or empty, default to wandb only (no trackio)
    if not hasattr(training_args, 'report_to') or not training_args.report_to:
        training_args.report_to = ["wandb"] if wandb_api_key else ["none"]
    elif isinstance(training_args.report_to, str):
        # Convert string to list, remove trackio
        report_list = [r.strip() for r in training_args.report_to.split(',')]
        report_list = [r for r in report_list if r and r.lower() != 'trackio']
        training_args.report_to = report_list if report_list else (["wandb"] if wandb_api_key else ["none"])
    elif isinstance(training_args.report_to, list):
        # Remove trackio from list
        training_args.report_to = [r for r in training_args.report_to if r and r.lower() != 'trackio']
        if not training_args.report_to:
            training_args.report_to = ["wandb"] if wandb_api_key else ["none"]
    
    # Set wandb run name to match output directory name
    if training_args.output_dir:
        # Extract just the directory name (not full path)
        run_name = os.path.basename(os.path.normpath(training_args.output_dir))
        training_args.run_name = run_name
        print(f"Setting wandb run_name to: {run_name}")
    
    # Set HuggingFace cache directory (optional - mainly for future Hub dataset downloads)
    # Note: With keep_in_memory=True in map operations, intermediate processing stays in memory
    # so dataset processing doesn't write to disk cache. This is mainly useful for:
    # - Future HuggingFace Hub dataset downloads
    # - General cache management
    # Models/tokenizers use HF_HOME/TRANSFORMERS_CACHE, not HF_DATASETS_CACHE
    cache_dir = script_args.hf_cache_dir or os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        print(f"Using HuggingFace datasets cache directory: {cache_dir}")
    # If not set, HuggingFace will use default location (~/.cache/huggingface/datasets)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        use_cache=False if training_args.gradient_checkpointing else True,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        logger.warning(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
        )

    ##############
    # Load dataset
    ##############
    # if (
    #     script_args.dataset_name
    #     and os.path.isfile(script_args.dataset_name)
    #     and script_args.dataset_name.endswith((".jsonl", ".json"))
    # ):
    #     # Local JSON/JSONL file
    #     dataset = load_dataset("json", data_files=script_args.dataset_name)
    # else:
    #     # Hugging Face Hub dataset name or local dataset directory/script
    #     dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    dataset_dir = script_args.dataset_name
    train_file = os.path.join(dataset_dir, "train.jsonl")
    test_file = os.path.join(dataset_dir, "test.jsonl")
    if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
        raise FileNotFoundError(
            f"Dataset files not found in {dataset_dir}"
        )
    
    logger.info(f"Loading dataset with keep_in_memory={training_args.keep_dataset_in_memory}")
    
    # Use the cache directory specified by user
    cache_dir = script_args.hf_cache_dir or os.getenv("HF_DATASETS_CACHE")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file,
        },
        keep_in_memory=training_args.keep_dataset_in_memory,
        cache_dir=cache_dir,  # Use the user-specified cache directory
    ) 

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
