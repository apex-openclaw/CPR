#!/usr/bin/env python3
"""Fast inference using vLLM for CPR evaluation.

Usage:
    python scripts/vllm_infer.py \
        --model Qwen/Qwen3-4B \
        --adapter outputs/qwen35-cpr-lora \
        --dataset data/prepared/cpr_test.jsonl \
        --output outputs/qwen35-cpr-lora/preds/generated_predictions.jsonl
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def build_prompt(example: dict, tokenizer) -> str:
    """Build chat prompt from alpaca-format example using Qwen3 template with thinking enabled."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    if inp:
        user_msg = f"{instruction}\n{inp}"
    else:
        user_msg = instruction
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    use_lora = args.adapter is not None
    # Safe cutoff: reserve room for output tokens so input + output <= max_model_len
    max_input_tokens = args.max_model_len - args.max_new_tokens

    examples = []
    with open(args.dataset) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [build_prompt(ex, tokenizer) for ex in examples]

    # Pre-filter prompts that exceed context length
    valid_indices = []
    skipped = 0
    for i, prompt in enumerate(prompts):
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens <= max_input_tokens:
            valid_indices.append(i)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} prompts exceeding {max_input_tokens} input tokens")

    valid_prompts = [prompts[i] for i in valid_indices]

    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    lora_request = None
    if use_lora:
        lora_request = LoRARequest("cpr_adapter", 1, args.adapter)

    print(f"Running inference on {len(valid_prompts)} prompts...")
    outputs = llm.generate(valid_prompts, sampling_params, lora_request=lora_request)

    # Reassemble results: valid predictions + empty placeholder for skipped
    result_map = {}
    for idx, output in zip(valid_indices, outputs):
        result_map[idx] = output.outputs[0].text

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in range(len(examples)):
            text = result_map.get(i, "")
            f.write(json.dumps({"predict": text}) + "\n")

    print(f"Saved {len(examples)} predictions ({skipped} skipped) to {out_path}")


if __name__ == "__main__":
    main()
