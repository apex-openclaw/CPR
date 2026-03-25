#!/usr/bin/env python3
"""Generate synthetic reasoning traces for CPR training data using a teacher model.

Uses vLLM for fast batch inference. The teacher model generates chain-of-thought
reasoning given the prompt + correct label, which can then be used for SFT.

Usage:
    python scripts/generate_reasoning_traces.py \
        --model Qwen/Qwen3-14B \
        --dataset train.jsonl \
        --output data/prepared/train_with_reasoning.jsonl \
        --tensor_parallel_size 2
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TRACE_PROMPT = """You are an expert in drug discovery, medicinal chemistry, and cell biology.

Given the following compound, its Cell Painting morphological profile, and the bioassay information, provide a detailed step-by-step reasoning for why this compound is {label} in this assay.

{original_prompt}

The correct answer is **PREDICTION: {label}**.

Now provide your detailed reasoning. Think step by step about:
1. The molecular structure (functional groups, scaffold, pharmacophore features) and likely pharmacological properties
2. What the morphological changes (elevated/reduced features) suggest about the compound's mechanism of action in cells
3. How the assay's biological target/pathway connects to the observed phenotypic changes
4. Why these observations are consistent with the compound being {label}

Write your reasoning, then conclude with exactly: **PREDICTION: {label}**"""


def extract_original_prompt(instruction: str) -> str:
    """Extract the core content (compound + features + bioassay) from the instruction."""
    # Remove the system preamble and instructions section, keep the data
    start_markers = ["## Compound", "## Task"]
    end_markers = ["## Instructions"]

    start = 0
    for marker in start_markers:
        idx = instruction.find(marker)
        if idx >= 0:
            start = idx
            break

    end = len(instruction)
    for marker in end_markers:
        idx = instruction.find(marker)
        if idx >= 0:
            end = idx
            break

    return instruction[start:end].strip()


def build_teacher_prompt(example: dict, tokenizer) -> str:
    """Build prompt for the teacher model to generate reasoning."""
    label = "ACTIVE" if "ACTIVE" in example["output"] and "INACTIVE" not in example["output"] else "INACTIVE"
    original_prompt = extract_original_prompt(example["instruction"])

    user_msg = TRACE_PROMPT.format(label=label, original_prompt=original_prompt)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ), label


def format_output_with_reasoning(reasoning: str, label: str) -> str:
    """Format the teacher's reasoning into training output format."""
    # Clean up the reasoning - extract content from <think> tags if present
    think_content = reasoning
    if "<think>" in reasoning and "</think>" in reasoning:
        think_start = reasoning.index("<think>") + len("<think>")
        think_end = reasoning.index("</think>")
        think_content = reasoning[think_start:think_end].strip()

    return f"<think>\n{think_content}\n</think>\n\n**PREDICTION: {label}**"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B",
                        help="Teacher model (Qwen3-14B for 2xH100, or Qwen3-32B)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Input training JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL with reasoning traces")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens for reasoning generation")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Process in batches (default: all at once)")
    args = parser.parse_args()

    # Load dataset
    examples = []
    with open(args.dataset) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {args.dataset}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build prompts
    prompts_and_labels = [build_teacher_prompt(ex, tokenizer) for ex in examples]
    prompts = [p for p, _ in prompts_and_labels]
    labels = [l for _, l in prompts_and_labels]

    # Filter by token length
    max_input_tokens = args.max_model_len - args.max_new_tokens
    valid_indices = []
    for i, prompt in enumerate(prompts):
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens <= max_input_tokens:
            valid_indices.append(i)
    skipped = len(examples) - len(valid_indices)
    if skipped:
        print(f"Skipped {skipped} prompts exceeding {max_input_tokens} input tokens")

    valid_prompts = [prompts[i] for i in valid_indices]
    print(f"Generating reasoning for {len(valid_prompts)} examples...")

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    # Generate in batches if specified
    if args.batch_size:
        all_outputs = []
        for start in range(0, len(valid_prompts), args.batch_size):
            batch = valid_prompts[start:start + args.batch_size]
            print(f"  Batch {start // args.batch_size + 1}: {len(batch)} prompts...")
            outputs = llm.generate(batch, sampling_params)
            all_outputs.extend(outputs)
    else:
        all_outputs = llm.generate(valid_prompts, sampling_params)

    # Build result map
    result_map = {}
    for idx, output in zip(valid_indices, all_outputs):
        reasoning = output.outputs[0].text
        label = labels[idx]
        result_map[idx] = format_output_with_reasoning(reasoning, label)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as f:
        for i, ex in enumerate(examples):
            if i in result_map:
                record = {
                    "instruction": ex["instruction"],
                    "input": ex.get("input", ""),
                    "output": result_map[i],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Saved {written} examples with reasoning to {out_path}")
    print(f"Skipped {len(examples) - written} examples")

    # Show a sample
    if result_map:
        first_idx = valid_indices[0]
        print(f"\n=== Sample output (example {first_idx}) ===")
        print(result_map[first_idx][:500])
        print("=== End sample ===")


if __name__ == "__main__":
    main()
