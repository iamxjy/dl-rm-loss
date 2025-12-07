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
accelerate launch \
    --config_file examples/accelerate_configs/single_gpu.yaml \
    grpo/grpo_modified.py \
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
    --log_completions \
    --reward_model_path reward_modeling/models_noise15_epoch1/Qwen2-0.5B-Reward-clean-BT/checkpoint-969/

"""

import os
from dataclasses import dataclass

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, set_seed, TrainerCallback

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

# remember to change the dataset prep to include more data


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_model_path: str = "Qwen2-0.5B-Reward-clean-BT/checkpoint-969/"
    use_debug_subset: bool = False
    debug_max_train_samples: int = 2
    debug_max_eval_samples: int = 2


class DDPStaticGraphCallback(TrainerCallback):
    """Callback to enable static graph in DDP before training starts."""
    
    def __init__(self, process_index):
        self.process_index = process_index
        self.already_set = False
    
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of the first training step, after model is fully wrapped."""
        if self.already_set or model is None or state.global_step > 0:
            return
        
        print(f"[Rank {self.process_index}] DDPStaticGraphCallback: Attempting to set static graph on first step...", flush=True)
        print(f"[Rank {self.process_index}] Model type: {type(model)}", flush=True)
        
        # The model should now be fully wrapped by DDP
        # Try to find and call _set_static_graph on the DDP wrapper
        if hasattr(model, '_set_static_graph'):
            print(f"[Rank {self.process_index}] Calling _set_static_graph() on model directly...", flush=True)
            model._set_static_graph()
            self.already_set = True
        elif hasattr(model, 'module'):
            print(f"[Rank {self.process_index}] Model has 'module' attribute, exploring...", flush=True)
            if hasattr(model.module, '_set_static_graph'):
                print(f"[Rank {self.process_index}] Calling _set_static_graph() on model.module...", flush=True)
                model.module._set_static_graph()
                self.already_set = True
            else:
                # Try going deeper for PEFT wrapped models
                current = model.module
                for depth in range(5):  # Try up to 5 levels deep
                    print(f"[Rank {self.process_index}] Checking depth {depth}, type: {type(current)}", flush=True)
                    if hasattr(current, '_set_static_graph'):
                        print(f"[Rank {self.process_index}] Found _set_static_graph() at depth {depth}, calling...", flush=True)
                        current._set_static_graph()
                        self.already_set = True
                        break
                    if hasattr(current, 'module'):
                        current = current.module
                    elif hasattr(current, 'model'):
                        current = current.model
                    else:
                        break
        
        if self.already_set:
            print(f"[Rank {self.process_index}] ✓ Successfully enabled static graph for DDP!", flush=True)
        else:
            print(f"[Rank {self.process_index}] WARNING: Could not enable static graph. Model type: {type(model)}", flush=True)
            if hasattr(model, 'module'):
                print(f"[Rank {self.process_index}] Model.module type: {type(model.module)}", flush=True)


# Disable TrackIO to avoid HuggingFace authentication issues
# os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")  # Disabled


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    state = PartialState()
    set_seed(training_args.seed)
    
    # Set HuggingFace cache directory to avoid disk quota issues in home directory
    hf_cache_dir = os.getenv("HF_HOME", "/proj/inf-scaling/iris/hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    print(f"[Rank {state.process_index}] Using HF cache: {hf_cache_dir}", flush=True)
    
    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        cache_dir=hf_cache_dir,
    )
    # In multi-process setups, Accelerate must handle device placement, not device_map
    if state.num_processes > 1:
        training_args.model_init_kwargs["device_map"] = None
    
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # device_map for quantization only applies to single-process
        if state.num_processes == 1:
            training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    # Set HuggingFace cache directory to avoid disk quota issues
    # Can be overridden by HF_DATASETS_CACHE environment variable
    cache_dir = os.getenv("HF_DATASETS_CACHE", "/proj/inf-scaling/iris/datasets_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    # Load UltraChat dataset - only rank 0 loads first to avoid simultaneous downloads
    try:
        if state.is_main_process:
            print(f"[Rank {state.process_index}] Loading dataset (main process)...", flush=True)
            dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen", cache_dir=cache_dir)
            print(f"[Rank {state.process_index}] Dataset loaded, size: {len(dataset)}", flush=True)
            
            # Create train/test split
            print(f"[Rank {state.process_index}] Creating train/test split...", flush=True)
            dataset = dataset.train_test_split(test_size=1000, seed=42)
            print(f"[Rank {state.process_index}] Split complete", flush=True)
        else:
            print(f"[Rank {state.process_index}] Waiting for main process to load dataset...", flush=True)
        
        # Wait for main process to finish loading
        print(f"[Rank {state.process_index}] About to wait for everyone...", flush=True)
        state.wait_for_everyone()
        print(f"[Rank {state.process_index}] Wait complete", flush=True)
        
        # Other processes now load from cache
        if not state.is_main_process:
            print(f"[Rank {state.process_index}] Loading dataset from cache...", flush=True)
            dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen", cache_dir=cache_dir)
            print(f"[Rank {state.process_index}] Creating train/test split...", flush=True)
            dataset = dataset.train_test_split(test_size=1000, seed=42)
            print(f"[Rank {state.process_index}] Dataset ready", flush=True)
        
        state.wait_for_everyone()
        print(f"[Rank {state.process_index}] All processes have loaded dataset", flush=True)
    except Exception as e:
        print(f"[Rank {state.process_index}] ERROR loading dataset: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # UltraChat has 'messages' field (chat format) which GRPO needs, not the string 'prompt' field
    # We need to convert messages to prompt format: extract all messages except the last assistant response
    def format_ultrachat_for_grpo(example):
        """Convert UltraChat messages to GRPO prompt format."""
        messages = example["messages"]
        # GRPO needs prompts in chat format (list of dicts), up to but not including the last assistant message
        # For generation, we take all messages except the final assistant response
        prompt_messages = []
        for msg in messages[:-1]:  # All messages except the last
            prompt_messages.append(msg)
        # Ensure we have at least a user message
        if not prompt_messages or prompt_messages[-1]["role"] != "user":
            # If last message isn't a user message, take all user messages
            prompt_messages = [msg for msg in messages if msg["role"] == "user"]
        return {"prompt": prompt_messages}
    
    print(f"[Rank {state.process_index}] Formatting dataset for GRPO...", flush=True)
    dataset = dataset.map(format_ultrachat_for_grpo, num_proc=1)
    print(f"[Rank {state.process_index}] Dataset formatted successfully", flush=True)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    # Reduce dataset to 1/15 of original size for faster training
    original_size = len(train_dataset)
    target_size = original_size // 15
    print(f"[Rank {state.process_index}] Reducing dataset from {original_size} to {target_size} samples (1/15 size)", flush=True)
    train_dataset = train_dataset.select(range(target_size))
    print(f"[Rank {state.process_index}] Dataset reduced successfully, new size: {len(train_dataset)}", flush=True)

    if script_args.use_debug_subset:
        train_dataset = train_dataset.select(
            range(min(script_args.debug_max_train_samples, len(train_dataset)))
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(
                range(min(script_args.debug_max_eval_samples, len(eval_dataset)))
            )

    ################
    # Training
    ################
    print(f"[Rank {state.process_index}] Starting trainer initialization...", flush=True)
    
    reward_model_path = os.path.expanduser(script_args.reward_model_path)
    
    # Models are pre-downloaded by the bash script before distributed launch
    # All ranks can safely load from cache using local_files_only
    print(f"[Rank {state.process_index}] Loading reward tokenizer from {reward_model_path} (local cache)...", flush=True)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_model_path,
        fix_mistral_regex=True,
        cache_dir=hf_cache_dir,
        local_files_only=True,  # Only use local cache, no downloads
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    print(f"[Rank {state.process_index}] Reward tokenizer loaded successfully", flush=True)
    
    # Load policy model tokenizer explicitly (Qwen3-0.6B-Base doesn't have a processor, only tokenizer)
    print(f"[Rank {state.process_index}] Loading policy model tokenizer from {model_args.model_name_or_path} (local cache)...", flush=True)
    policy_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=hf_cache_dir,
        local_files_only=True,  # Only use local cache, no downloads
    )
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    print(f"[Rank {state.process_index}] Policy tokenizer loaded successfully", flush=True)
    
    print(f"[Rank {state.process_index}] Initializing GRPOTrainer with policy model: {model_args.model_name_or_path}", flush=True)
    print(f"[Rank {state.process_index}] Reward model path: {reward_model_path}", flush=True)
    
    # Force local files only to prevent any downloads during trainer init
    training_args.model_init_kwargs["local_files_only"] = True
    
    # Enable DDP static graph mode to fix gradient checkpointing + DDP + LoRA incompatibility
    if state.num_processes > 1:
        print(f"[Rank {state.process_index}] Configuring DDP with static_graph=True...", flush=True)
        # Set DDP kwargs through training args
        if not hasattr(training_args, 'ddp_find_unused_parameters'):
            training_args.ddp_find_unused_parameters = False
        if not hasattr(training_args, 'ddp_bucket_cap_mb'):
            training_args.ddp_bucket_cap_mb = 25
        # Note: ddp_static_graph is not directly supported in transformers TrainingArguments
        # We'll use the callback approach as a fallback
        print(f"[Rank {state.process_index}] DDP settings: find_unused_parameters={training_args.ddp_find_unused_parameters}", flush=True)
    
    # Add callback to enable DDP static graph (fixes gradient checkpointing + DDP + LoRA issue)
    ddp_callback = DDPStaticGraphCallback(state.process_index) if state.num_processes > 1 else None

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[reward_model_path],
        reward_processing_classes=[reward_tokenizer],
        processing_class=policy_tokenizer,  # Explicitly pass tokenizer for policy model
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[ddp_callback] if ddp_callback else [],
    )
    
    print(f"[Rank {state.process_index}] GRPOTrainer initialized successfully!", flush=True)
    
    # Monkey-patch the accelerator's prepare method to enable static graph after DDP wrapping
    if state.num_processes > 1:
        print(f"[Rank {state.process_index}] Setting up DDP static graph hook...", flush=True)
        original_prepare = trainer.accelerator._prepare_one
        
        def patched_prepare(obj, *args, **kwargs):
            result = original_prepare(obj, *args, **kwargs)
            # After preparing (wrapping with DDP), try to enable static graph
            if hasattr(result, '_set_static_graph') and not getattr(result, '_static_graph_set', False):
                print(f"[Rank {state.process_index}] ✓ Enabling static graph on DDP-wrapped model!", flush=True)
                result._set_static_graph()
                result._static_graph_set = True  # Mark so we don't call it again
            return result
        
        trainer.accelerator._prepare_one = patched_prepare
        print(f"[Rank {state.process_index}] DDP static graph hook installed", flush=True)

    trainer.train()

    # Save on main process only (hub push disabled)
    accelerator = getattr(trainer, "accelerator", None)
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    if is_main_process:
        trainer.save_model(training_args.output_dir)

    if accelerator is not None:
        accelerator.wait_for_everyone()
