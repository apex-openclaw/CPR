# CPR — Cell Painting Reasoning

LLM-based reasoning pipeline for Cell Painting bioassays. The goal is to teach a
Qwen 3.5 4B model to read CellProfiler feature summaries + SMILES + assay
metadata, reason about the phenotype, and output **PREDICTION: ACTIVE/INACTIVE**
per assay. SFT is handled via [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory);
reinforcement learning can be applied later on top of the SFT checkpoint.

## Repository layout

```
data_prep/            # dataset_config + prepare_cpr_dataset.py
configs/              # LlamaFactory configs for SFT / inference
scripts/              # helper shell scripts
requirements.txt      # minimal Python deps (llamafactory + pandas/numpy/...)
```

## Prerequisites

- Python ≥ 3.10
- NVIDIA GPU with ≥48 GB memory (e.g., RTX 6000 Ada). The Mac mini cannot finetune this model.
- Hugging Face login (for pulling Qwen checkpoints) and `pip install -r requirements.txt`.

## 1. Data preparation

1. Copy the cpg0012 resources (labels CSV, CellProfiler NPZ, split CSVs, assay descriptions, optional DeepSeek traces) onto your GPU server.
2. Edit `data_prep/dataset_config.yaml`:
   - Update `data_root` / individual `paths` to match your filesystem.
   - Point `assay_desc_json`/`traces_dir` to whichever reference repo (e.g., `MFCP-Reasoning`).
   - Adjust prompt filters (top-K features, assay include list, per-assay caps, etc.).
3. Run the prep script (defaults to the YAML above):
   ```bash
   cd CPR
   scripts/run_data_prep.sh
   ```
4. Outputs land in `data/prepared/`:
   - `train.jsonl`, `val.jsonl`, `test.jsonl` (Alpaca format `{"instruction", "input", "output"}`)
   - Aliases `cpr_train.jsonl`, `cpr_val.jsonl`, `cpr_test.jsonl` for LlamaFactory
   - `metadata.csv` listing (split, assay_id, inchikey, label, smiles)

If curated traces exist (DeepSeek-generated reasoning that predicted correctly),
the script uses them for the train split. Otherwise, it backfills a deterministic
template reasoning that still ends with `**PREDICTION: ...**`.

## 2. Supervised finetuning (SFT)

1. Review `configs/sft_qwen35.yaml`:
   - Replace `model_name_or_path` with the actual Qwen 3.5 4B checkpoint (placeholder currently uses Qwen2.5). Ensure the tokenizer template (`qwen2`) still matches.
   - Adjust LoRA / batch / epochs as needed for your GPU budget.
   - `dataset` references `cpr_train` and `cpr_val` produced above.
2. Kick off training:
   ```bash
   CUDA_VISIBLE_DEVICES=0 scripts/run_sft.sh
   ```
   LlamaFactory will save LoRA weights + logs under `outputs/qwen35-cpr-lora/`.

## 3. Inference / evaluation

1. Update `configs/infer_qwen35.yaml` so `model_name_or_path` / `adapter_name_or_path` point to your SFT artifacts.
2. Run:
   ```bash
   CUDA_VISIBLE_DEVICES=0 scripts/run_infer.sh
   ```
   Predictions (reasoning + `**PREDICTION: …**`) are written to `outputs/qwen35-cpr-lora/preds/`.
3. Use `data/prepared/metadata.csv` to join predictions with ground-truth labels and compute per-assay metrics (scikit-learn scripts can live in `notebooks/` or a future `eval/` folder).

## Notes & next steps

- The configs default to QLoRA (4-bit) to keep memory <48 GB; adjust to full-precision if you move to A100s.
- Reinforcement learning (per Joshua’s plan) can reuse the same data/prep outputs—the RL script just needs a directory of reasoning traces + reward shaping.
- See [`hwjustin/MFCP-Reasoning`](https://github.com/hwjustin/MFCP-Reasoning) for reference prompt templates and DeepSeek trace generation.
