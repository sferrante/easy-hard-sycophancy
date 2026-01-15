# Fine-tuning Gemma Away From Sycophancy

This project studies **sycophancy reduction** via fine-tuning on paired-choice prompts.  
I compare **full supervised fine-tuning (SFT)** vs **LoRA** on a Gemma instruction model, and evaluate whether the model prefers the **sycophantic** answer or the **non-sycophantic** answer using a simple A/B log-prob margin.

---

## Overview

- **Goal:** reduce sycophancy (the tendency to overly agree with a user’s stated beliefs) by training on examples with a “sycophantic” vs “non-sycophantic” option.
- **Models:**
  - **Base:** original Gemma instruction model (no fine-tuning)
  - **SFT:** full supervised fine-tune
  - **LoRA:** parameter-efficient fine-tune
- **Evaluation:** compute a preference margin  
  \[
  \Delta = \log P(\text{Syco}) - \log P(\text{Non-Syco})
  \]
  where **Δ > 0** means the model prefers the **sycophantic** option.

I also test for **A/B position bias** (models sometimes learn “pick (A)” or “pick (B)” regardless of content) using a swap-based evaluation.

---

## Contents

- `train.ipynb`  
  Trains **SFT** and **LoRA** variants on the easy split (HuggingFace + Transformers Trainer).  
  Includes training configuration, logging, and checkpoint saving.

- `eval_and_plots.ipynb`  
  Evaluates base/SFT/LoRA on **easy** and **hard** splits.  
  Computes Δ margins and generates the plots below, including a **swap-avg** evaluation to reduce A/B position artifacts.

---

## Setup

Recommended environment:

```bash
pip install torch transformers datasets accelerate peft bitsandbytes tqdm matplotlib
