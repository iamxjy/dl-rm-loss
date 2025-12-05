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
    keep_dataset_in_memory: bool = field(
        default=True,
        metadata={
            "help": "Whether to keep the dataset in memory after loading. If True, dataset stays in memory. "
            "If False, dataset is written to disk cache."
        },
    )


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = HfArgumentParser((RewardModelingScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Pass keep_dataset_in_memory flag to training args
    training_args.keep_dataset_in_memory = script_args.keep_dataset_in_memory
    
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
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file,
        },
        keep_in_memory=script_args.keep_dataset_in_memory,
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
