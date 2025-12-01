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
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# For Qwen/Qwen2.5-3B-Instruct (text-only model)
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_test.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir grpo-Qwen2.5-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions

"""

import os

import torch
from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import get_soft_overlong_punishment


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    # Load UltraChat dataset
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
    
    # Create train/test split
    dataset = dataset.train_test_split(test_size=1000, seed=42)

    def format_prompt(example):
        """
        Convert UltraChat messages to prompt-only format for GRPO.
        UltraChat has 'messages' field with conversation history.
        We extract all messages up to (but not including) the last assistant message as the prompt.
        """
        messages = example["messages"]
        
        # Extract prompt: all messages except the last assistant response
        # For GRPO, we want the model to generate the completion
        prompt_messages = []
        for msg in messages[:-1]:  # All but the last message
            prompt_messages.append(msg)
        
        return {"prompt": prompt_messages}

    dataset = dataset.map(format_prompt)

    '''
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None
    '''
    train_dataset = dataset["train"].select(range(min(2, len(dataset["train"]))))
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset["test"].select(range(min(2, len(dataset["test"]))))

    ################
    # Training
    ################
    # Create soft overlong punishment reward function
    # This penalizes completions that are too long, which is useful for general text generation
    # max_completion_length should match the training config (1024 in this case)
    soft_overlong_punishment = get_soft_overlong_punishment(
        max_completion_len=training_args.max_completion_length or 1024,
        soft_punish_cache=256,  # Soft penalty zone: 256 tokens before max length
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[soft_overlong_punishment],  # Use soft_overlong_punishment for general text generation
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
