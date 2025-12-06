"""Compare two models using OpenAI as a pairwise judge.

Run:
    python judging/compare_llms_with_openai_judge.py \
        --model-a Qwen/Qwen2-0.5B-Instruct \
        --model-b Qwen/Qwen3-0.6B \
        --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.judges import OpenAIPairwiseJudge


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two HF models with an OpenAI LLM judge.")
    parser.add_argument(
        "--model-a",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HF model id or local path for candidate A.",
    )
    parser.add_argument(
        "--model-b",
        default="Qwen/Qwen3-0.6B",
        help="HF model id or local path for candidate B.",
    )
    parser.add_argument(
        "--arena-hard",
        action="store_true",
        help="Use Arena-Hard-Auto v2.0 prompts.",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model id for judging.")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit number of prompts evaluated (if provided).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=str,
        nargs="?",
        const="judging/results/per_prompt.jsonl",
        default="judging/results/per_prompt.jsonl",
        help="Path to write per-prompt records as JSONL (defaults to judging/results/per_prompt.jsonl when omitted).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        nargs="?",
        const="judging/results/summary.csv",
        default="judging/results/summary.csv",
        help="Path to write a CSV summary (defaults to judging/results/summary.csv when omitted).",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, device: str):
    model_id = os.path.expanduser(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Some tiny models may not have a pad token; fall back to eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    config: GenerationConfig,
) -> list[str]:
    if isinstance(prompts, str):
        prompts = [prompts]

    if getattr(tokenizer, "chat_template", None):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature if config.temperature > 0 else None,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generations: list[str] = []
    for prompt, output_ids in zip(prompts, outputs, strict=True):
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        # Best effort to strip the prompt prefix.
        if full_text.startswith(prompt):
            generations.append(full_text[len(prompt) :].strip())
        else:
            generations.append(full_text.strip())
    return generations


def load_arena_hard_prompts() -> list[str]:
    """Load Arena-Hard-Auto v2.0 prompts from question.jsonl via datasets."""
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Only load the v2.0 question file and keep streaming to tolerate schema variety.
    ds: Iterable[dict] = load_dataset(
        "lmarena-ai/arena-hard-auto",
        data_files="data/arena-hard-v2.0/question.jsonl",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    def extract_prompt(row: dict) -> str | None:
        # 1) messages/turns-style schemas
        for key in ("messages", "turns"):
            seq = row.get(key)
            if isinstance(seq, list) and seq:
                first = seq[0]
                if isinstance(first, dict):
                    content = first.get("content") or first.get("text")
                    if isinstance(content, str):
                        return content
                    # If content is a list of segments, consider joining them.
                elif isinstance(first, str):
                    return first

        # 2) Simple text fields
        for key in ("prompt", "question", "instruction", "user"):
            v = row.get(key)
            if isinstance(v, str):
                return v

        return None

    iterator = iter(ds)
    try:
        first_row = next(iterator)
    except StopIteration:
        raise ValueError("No rows found in Arena-Hard-Auto v2.0 question.jsonl.")

    prompts: list[str] = []
    first_prompt = extract_prompt(first_row)
    if first_prompt:
        prompts.append(first_prompt)

    for row in iterator:
        prompt = extract_prompt(row)
        if prompt:
            prompts.append(prompt)

    if not prompts:
        raise ValueError("No prompts extracted from Arena-Hard-Auto v2.0 question.jsonl.")

    return prompts


def _prepare_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_jsonl_per_prompt(
    path: str,
    prompts: Sequence[str],
    pairs: Sequence[Sequence[str]],
    ranks: Sequence[int],
    run_metadata: dict,
) -> None:
    """Write one JSON object per prompt with full completions and judge pick."""
    p = _prepare_path(path)
    with p.open("w", encoding="utf-8") as f:
        for idx, (prompt, pair, rank) in enumerate(zip(prompts, pairs, ranks, strict=True)):
            winner = (
                run_metadata["model_a"]
                if rank == 0
                else run_metadata["model_b"]
                if rank == 1
                else None
            )
            record = {
                **run_metadata,
                "prompt_idx": idx,
                "prompt": prompt,
                "completion_a": pair[0],
                "completion_b": pair[1],
                "judge_pick": rank,
                "winner_model": winner,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv_summary(
    path: str,
    prompts: Sequence[str],
    ranks: Sequence[int],
    run_metadata: dict,
    wins_a: int,
    wins_b: int,
) -> None:
    """Write a lightweight CSV: per-prompt winner plus a totals row."""
    p = _prepare_path(path)
    fieldnames = [
        "prompt_idx",
        "prompt",
        "judge_pick",
        "winner_model",
        "model_a",
        "model_b",
        "judge_model",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (prompt, rank) in enumerate(zip(prompts, ranks, strict=True)):
            winner = (
                run_metadata["model_a"]
                if rank == 0
                else run_metadata["model_b"]
                if rank == 1
                else None
            )
            writer.writerow(
                {
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "judge_pick": rank,
                    "winner_model": winner,
                    "model_a": run_metadata["model_a"],
                    "model_b": run_metadata["model_b"],
                    "judge_model": run_metadata["judge_model"],
                }
            )
        writer.writerow(
            {
                "prompt_idx": "total",
                "prompt": "",
                "judge_pick": "",
                "winner_model": f"{wins_a} vs {wins_b}",
                "model_a": run_metadata["model_a"],
                "model_b": run_metadata["model_b"],
                "judge_model": run_metadata["judge_model"],
            }
        )


def main() -> None:
    args = parse_args()
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Please set OPENAI_API_KEY in the environment.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a_id = os.path.expanduser(args.model_a)
    model_b_id = os.path.expanduser(args.model_b)

    print(f"Using device: {device}")
    print(f"Model A: {model_a_id}")
    print(f"Model B: {model_b_id}")
    print(f"Judge model: {args.judge_model}")

    if args.arena_hard:
        prompts: Sequence[str] = load_arena_hard_prompts()
    else:
        prompts: Sequence[str] = (
            "Explain gravity like I'm 12.",
            "Give me a 1-sentence summary of the French Revolution.",
            "List three creative ice cream flavors.",
            "Write a haiku about the ocean.",
        )
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    gen_config = GenerationConfig()
    model_a, tok_a = load_model_and_tokenizer(model_a_id, device)
    model_b, tok_b = load_model_and_tokenizer(model_b_id, device)

    run_metadata = {
        "model_a": model_a_id,
        "model_b": model_b_id,
        "judge_model": args.judge_model,
        "generation": asdict(gen_config),
        "num_prompts": len(prompts),
    }

    print(f"Generating completions for {len(prompts)} prompts...")
    completions_a: list[str] = []
    completions_b: list[str] = []

    def batched(seq: Sequence[str], batch_size: int):
        for idx in range(0, len(seq), batch_size):
            yield seq[idx : idx + batch_size]

    for prompt_batch in batched(list(prompts), args.batch_size):
        completions_a.extend(generate(model_a, tok_a, prompt_batch, device, gen_config))
        completions_b.extend(generate(model_b, tok_b, prompt_batch, device, gen_config))

    pairs = [list(pair) for pair in zip(completions_a, completions_b, strict=True)]

    judge = OpenAIPairwiseJudge(model=args.judge_model, max_requests=1_000)
    print("Querying judge (0 means first response wins, 1 means second)...")
    ranks = judge.judge(prompts=list(prompts), completions=pairs, shuffle_order=True)

    wins_a = sum(rank == 0 for rank in ranks)
    wins_b = sum(rank == 1 for rank in ranks)
    if args.jsonl_out:
        save_jsonl_per_prompt(args.jsonl_out, prompts, pairs, ranks, run_metadata)
        print(f"Wrote per-prompt JSONL to {args.jsonl_out}")
    if args.csv_out:
        save_csv_summary(args.csv_out, prompts, ranks, run_metadata, wins_a, wins_b)
        print(f"Wrote CSV summary to {args.csv_out}")
    total = len(prompts)
    ties = total - wins_a - wins_b
    print(
        f"Final summary: {model_a_id} wins {wins_a}/{total}, "
        f"{model_b_id} wins {wins_b}/{total}, ties {ties}."
    )


if __name__ == "__main__":
    main()

