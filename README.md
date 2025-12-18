
# AdaRank: Adaptive Rank Fine-Tuning for LoRA (MRPC Experiment)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/HuggingFace_Transformers-4.45+-orange)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)

**AdaRank** is an experimental implementation of adaptive-rank LoRA fine-tuning, inspired by the theoretical framework from the paper *"AdpRank: Adaptive Rank Fine-Tuning through Controlled Overparameterization and Double-Descent Dynamics (My Underdrad Thesis)"*.

This repository contains a lightweight, working prototype tested specifically on the **MRPC** (Microsoft Research Paraphrase Corpus) task from GLUE using **T5-base** as the backbone. The goal is to demonstrate the core idea: dynamically increasing LoRA rank during training when loss plateaus, allowing temporary overparameterization to exploit potential generalization benefits from the "second descent" regime, followed by continued training at the higher rank.

> **Note**: This is an **early experimental version**. While the full paper proposes inflation → overparameterized descent → pruning, this notebook currently implements **rank inflation only** (with fresh optimizer reset) on MRPC. Pruning and other datasets are planned for future iterations.

## Key Features

- Pure PyTorch + Hugging Face `peft` implementation of dynamic LoRA rank adjustment
- Custom `TrainerCallback`-based controller (with fixes for common Accelerate scaler issues)
- Alternative pure-PyTorch training loop with `tqdm` (recommended for stability)
- Proper text-to-text formatting for MRPC using T5-style prompts
- Accurate metric computation (Accuracy + macro F1) with robust label decoding
- Logging of rank changes, training loss, and validation metrics
- Easy configuration via `AdaRankCfg` dataclass

## Installation

This notebook was tested on Kaggle (GPU P100) with Python 3.11.

```bash
pip install -U datasets accelerate evaluate tabulate protobuf \
    transformers peft torch matplotlib seaborn pandas scipy
```

> Note: Some dependency conflicts may appear (e.g., protobuf version), but the core libraries work correctly.

## Dataset

- **Task**: MRPC (paraphrase identification) from GLUE
- **Input format**: T5-style prompt  
  `"mrpc sentence1: {sentence1} sentence2: {sentence2} paraphrase:"`
- **Targets**: `"equivalent"` or `"not equivalent"`
- Train / Validation splits used directly from `datasets.load_dataset("glue", "mrpc")`

## Model

- Backbone: `t5-base` (Seq2Seq LM)
- PEFT method: LoRA applied to query and value projections (`target_modules=["q", "v"]`)
- Initial rank: 4 (configurable)
- Max rank: 64–128 (configurable)
- Scaling: `lora_alpha = 2 * rank`

## AdaRank Controller

The adaptive logic is implemented in two ways:

1. **TrainerCallback version** (uses Hugging Face `Seq2SeqTrainer`)
2. **Pure PyTorch version** (recommended – more stable when changing rank)

### How Rank Inflation Works

- Monitor training loss every N steps
- If loss > tolerance and warmup complete → increase rank by `alpha`
- When rank increases:
  - Merge current LoRA into base model
  - Create new higher-rank LoRA adapter
  - Replace model in trainer / loop
  - **Create fresh optimizer and scheduler** (critical for continued learning)
  - (In callback version) Re-create `GradScaler` to avoid NaN/inf issues

### Configuration (AdaRankCfg)

```python
AdaRankCfg(
    r0=4,           # starting rank
    r_max=64,       # maximum allowed rank
    alpha=2,        # rank increase step
    tol=0.10,       # loss plateau threshold
    check_every=50, # how often to check loss
    warmup_steps=50 # steps before starting adaptation
)
```

## Results (Typical Run on MRPC)

With the pure PyTorch loop and reasonable settings:

- Starts at rank 4
- Inflates to higher ranks (e.g., 4 → 6 → 8 → ... up to ~32–48) when loss stalls
- Final validation performance typically reaches **~85–87% Accuracy / ~89–91% F1**
- Often matches or slightly exceeds fixed-rank LoRA (r=16 or 32) while using less average rank early on

Exact numbers vary by seed and hyperparameters.

## Usage

1. Open the notebook `adp_mrpc.ipynb`
2. Choose either:
   - The `Seq2SeqTrainer` + callback version (may hit scaler issues)
   - The **pure PyTorch loop** (cell near the end) – more reliable
3. Adjust `AdaRankCfg` and training args as needed
4. Run all cells

## Current Limitations & Future Work

- Only tested on **MRPC** – other GLUE tasks, vision, or larger models (LLaMA, etc.) pending
- No final **SVD pruning** phase yet (only inflation)
- No layer-specific rank adaptation
- No comparison tables vs. fixed-rank baselines (manual runs needed)
- Hyperparameters (tol, alpha, check frequency) are sensitive and may need tuning per task

Planned extensions:
- Add pruning after overparameterized phase
- Support RoBERTa-base classification setup
- Multi-task / GLUE suite evaluation
- Visualization of double-descent curve (test error vs. effective rank over time)

## References

- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
- Zhang et al. (2023). AdaLoRA
- Valipour et al. (2023). DyLoRA
- Proposed method inspired by: *AdpRank: Adaptive Rank Fine-Tuning through Controlled Overparameterization and Double-Descent Dynamics* (2025)

## License

MIT License – feel free to use, modify, and extend for research purposes.

---

**Happy experimenting!**  
If you find interesting double-descent behavior or improved results on other tasks, contributions and reports are welcome.
```
