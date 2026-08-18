# LLM-Finetuning

A hands-on, notebook-based walkthrough of the core techniques for running and **fine-tuning large language models (LLMs)** on modest hardware, using Llama 3.2 and the Hugging Face ecosystem (`transformers`, `bitsandbytes`, `peft`, `trl`).

The two flagship notebooks answer two practical questions:

1. **How do I fit a large model into limited GPU memory?** → [`1-Quantization.ipynb`](1-Quantization.ipynb)
2. **How do I teach a model a new skill cheaply, without retraining all of its weights?** → [`2-InstructVsBaseModel.ipynb`](2-InstructVsBaseModel.ipynb)

Every notebook is designed to run on a **GPU runtime** (e.g. Google Colab) and authenticates to the Hugging Face Hub with a token read from a local `.env` file.

---

## Table of contents

- [Key concepts](#key-concepts)
  - [1. Quantization](#1-quantization)
  - [2. Tokenization — and why the tokenizer is *not* fine-tuned](#2-tokenization--and-why-the-tokenizer-is-not-fine-tuned)
  - [3. Base vs. instruct models & chat templates](#3-base-vs-instruct-models--chat-templates)
  - [4. Fine-tuning and SFT](#4-fine-tuning-and-sft)
  - [5. LoRA (Low-Rank Adaptation)](#5-lora-low-rank-adaptation)
- [The notebooks](#the-notebooks)
- [Setup](#setup)
- [Requirements](#requirements)

---

## Key concepts

### 1. Quantization

A neural network is just a large collection of numbers (**weights**). By default those numbers are stored as 32-bit floating point (`float32`), which takes **4 bytes per weight**. A 1-billion-parameter model therefore needs ~4 GB *just to hold the weights* — before you add activations, gradients, or optimizer state.

**Quantization** reduces the number of bits used to store each weight. Fewer bits → less memory and faster memory transfers, at the cost of some numerical precision.

| Precision | dtype | Bytes / weight | ~Size of a 1.24B model | Typical use |
|-----------|-----------|----------------|------------------------|-------------|
| 32-bit | `float32` | 4 | ~4.6 GB | Full precision / training reference |
| 16-bit | `bfloat16` / `float16` | 2 | ~2.3 GB | Standard for training & inference |
| 8-bit | `int8` (LLM.int8) | ~1 | ~1.4 GB | Memory-efficient inference / LoRA training |
| 4-bit | `NF4` (QLoRA) | ~0.5 | ~0.9 GB | Maximum compression |

**How does 8-bit / 4-bit work?** Instead of storing a real number directly, the library (`bitsandbytes`) stores a small integer plus a scaling factor per block of weights. At compute time the integers are de-quantized back to a float, multiplied, and the result accumulated in higher precision. Schemes like **LLM.int8()** keep the rare large-magnitude "outlier" features in 16-bit so accuracy barely changes, while **NF4** (used in QLoRA) uses a data-type tuned for the bell-curve distribution of neural-network weights.

**Why it matters for fine-tuning:** quantization is what lets us load a model in 8-bit and *still* fine-tune it on a single small GPU — because only the tiny LoRA adapters (see below) need full-precision gradients, while the big frozen base stays compressed.

You control this in code with a single config object:

```python
from transformers import BitsAndBytesConfig
config_8bit = BitsAndBytesConfig(load_in_8bit=True)          # 8-bit
config_4bit = BitsAndBytesConfig(load_in_4bit=True,          # 4-bit (QLoRA-style)
                                 bnb_4bit_quant_type="nf4")
```

### 2. Tokenization — and why the tokenizer is *not* fine-tuned

This is a point that trips up almost everyone, so it deserves its own section.

A model cannot read text; it reads **token IDs** (integers). The **tokenizer** is the component that converts text ↔ IDs:

```
"list docker containers"  ──tokenizer──▶  [1214, 27686, 18021, 7086]  ──model──▶  predictions
```

There are **two completely separate "training" processes**, and they are easy to confuse:

| | **Training the tokenizer** | **Fine-tuning the model** |
|---|---|---|
| **What is learned** | The **vocabulary** and merge rules — which character sequences become which tokens (e.g. via Byte-Pair Encoding / BPE) | The model's **weights** — the billions of numbers that turn token IDs into predictions |
| **Data used** | A large raw text corpus, analyzed for frequent sub-word patterns | Labelled examples of the task (input → desired output) |
| **When it happens** | **Once, upstream**, by the model's original authors, before the model is ever trained | Later, by *you*, to adapt a pretrained model to a task |
| **In these notebooks** | **Not done.** We *load* the pretrained tokenizer with `AutoTokenizer.from_pretrained(...)` and reuse it unchanged | **This is the whole point of notebook 2** — we update (a small slice of) the weights |

**Key takeaway:** in [`2-InstructVsBaseModel.ipynb`](2-InstructVsBaseModel.ipynb) the tokenizer is **frozen and reused as-is**. We never re-learn the vocabulary. The tokenizer only *does three jobs* for us:

1. **Encodes** our text into `input_ids` + `attention_mask`.
2. **Applies the chat template** — it knows the special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) that turn a list of `system`/`user`/`assistant` messages into the exact string format the instruct model expects.
3. **Handles padding** — Llama has no dedicated pad token, so we tell the tokenizer to reuse the end-of-sequence token (`tokenizer.pad_token = tokenizer.eos_token`).

So when notebook 2 "does tokenization," it is **using** the tokenizer, not **training** it. All the actual learning during fine-tuning happens in the model's LoRA weights. (You *would* re-train or extend a tokenizer only if your data used a language, alphabet, or domain vocabulary the original tokenizer handles badly — that is a different, rarer task and is out of scope here.)

### 3. Base vs. instruct models & chat templates

LLMs come in two common flavors, and knowing the difference is the key to using them correctly:

| Aspect | **Base model** | **Instruct model** |
|--------|----------------|--------------------|
| **Training** | Pretrained only — predicts the next token on raw internet-scale text | Base model **+** instruction tuning (SFT / RLHF) |
| **Behaviour** | *Continues* / completes text; does not "follow" requests | Follows instructions, answers questions, holds a chat |
| **Input format** | Plain text | A **chat template**: `system` / `user` / `assistant` turns wrapped in special tokens |

An instruct model is trained to expect a **chat template** built from model-specific special tokens. For Llama 3.2 a formatted conversation looks like this:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
...system prompt...<|eot_id|><|start_header_id|>user<|end_header_id|>
...user message...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Different model families use different tokens (Mistral uses `[INST] ... [/INST]` with `<s>`/`</s>`). A **base** model has never seen any of these tokens, so feeding it a chat template works poorly — and feeding a raw prompt to an instruct model wastes its training. Notebook 2 renders the *same* conversation with both the Mistral and Llama templates to make this contrast concrete.

### 4. Fine-tuning and SFT

**Fine-tuning** = taking a model that already knows a lot (from pretraining) and nudging its weights so it performs a *specific* task well.

**Supervised Fine-Tuning (SFT)** is the most common recipe: you give the model input→output examples, it predicts the next token at every position, and the loss measures how wrong those predictions are. Gradient descent then updates the weights to reduce that loss. In notebook 2 the "output" the model learns to produce is a **Docker command** for a given English request.

Two supporting mechanics you'll see:

- **Data collator & dynamic padding** — sequences have different lengths, so `DataCollatorForLanguageModeling(mlm=False)` pads each *batch* to its own longest sequence and builds the shifted `labels` automatically (causal LM = predict the next token).
- **Label masking** — padded positions get a label of `-100` so the loss function **ignores** them; the model is never rewarded or penalized for what it "predicts" on padding.

### 5. LoRA (Low-Rank Adaptation)

Fine-tuning **every** weight of a 1.24-billion-parameter model is expensive: you must store gradients and optimizer state for *all* of them, and you end up with a full-size copy of the model for every task. **LoRA** sidesteps this.

**The idea.** For a pretrained weight matrix `W` (shape `d × d`), LoRA keeps `W` **frozen** and learns a small *low-rank* update instead of modifying `W` directly:

```
ΔW = B · A       where   A ∈ ℝ^(r × d),   B ∈ ℝ^(d × r),   r ≪ d
```

so a layer's output becomes:

```
h = W·x + (α / r) · B · A · x
```

Only the small matrices `A` and `B` are trained; `W` never changes. Because the rank `r` is tiny compared to `d`, the number of trainable parameters collapses — in notebook 2, from **~1.24B down to ~45M (~3.5%)**.

**Why we use LoRA here:**

- **It fits on a small GPU.** Combined with 8-bit quantization, only the tiny adapters need gradients and optimizer state.
- **It is faster and cheaper** to train than full fine-tuning.
- **Adapters are tiny and portable** — a saved adapter is a few MB, so you can keep one base model and hot-swap task-specific adapters.
- **No catastrophic forgetting** — the base weights are untouched, so the model keeps its general abilities.
- **Quality stays competitive** with full fine-tuning on many downstream tasks.

The LoRA knobs (set via `LoraConfig`):

| Parameter | What it controls |
|-----------|------------------|
| `r` | Rank of the update — higher = more capacity and more parameters (notebook uses `64`) |
| `lora_alpha` | Scaling factor `α`; the update is scaled by `α / r` |
| `target_modules` | Which layers get adapters (attention `q/k/v/o` and MLP `gate/up/down` projections) |
| `lora_dropout` | Dropout on the LoRA path, for regularization |
| `bias`, `task_type` | Whether to train biases; the task family (`CAUSAL_LM`) |

After training you can **merge** the adapters back into the base weights (`merge_and_unload()`), which folds `W ← W + (α/r)·B·A` and gives you a standalone fine-tuned model with no LoRA wrappers left.

---

## The notebooks

### 1. [`1-Quantization.ipynb`](1-Quantization.ipynb) — Model quantization

Loads the same model, `meta-llama/Llama-3.2-1B-Instruct`, at **four precisions (32/16/8/4-bit)** and compares them side by side. You will see how to:

- Load a model in 4-, 8-, 16-, and 32-bit with `BitsAndBytesConfig`.
- Measure and compare each variant's memory footprint (see the size table [above](#1-quantization)).
- Inspect how the raw weight values change (floating-point vs. integer) after quantization.
- See how the layer types differ per precision (`Linear`, `Linear8bitLt`, `Linear4bit`).
- Measure real GPU memory usage for each variant.

This notebook builds the intuition for *why* quantization is what makes the fine-tuning in notebook 2 possible on modest hardware.

### 2. [`2-InstructVsBaseModel.ipynb`](2-InstructVsBaseModel.ipynb) — Instruct vs. base models & LoRA fine-tuning

A complete, end-to-end fine-tuning walkthrough. It first makes the [base-vs-instruct distinction](#3-base-vs-instruct-models--chat-templates) concrete, then fine-tunes `meta-llama/Llama-3.2-1B-Instruct` to translate plain-English requests into **Docker commands**.

**The full pipeline, step by step:**

1. **Authenticate** to the Hugging Face Hub using the token from your `.env` file.
2. **Load the instruct model in 8-bit** (`BitsAndBytesConfig(load_in_8bit=True)`) so it fits on a small GPU. Inspecting the model shows its projections are now `Linear8bitLt` layers.
3. **Load the tokenizer** (`AutoTokenizer.from_pretrained`) — *loaded, not trained* (see [tokenizer vs. fine-tuning](#2-tokenization--and-why-the-tokenizer-is-not-fine-tuned)). `padding_side="left"` is set because it matters for generation.
4. **Load & split the dataset** [`MattCoddity/dockerNLcommands`](https://huggingface.co/datasets/MattCoddity/dockerNLcommands) into an 80/20 train/validation split. Each row has an `instruction`, an `input` (the English request), and an `output` (the target Docker command).
5. **Format each row as a chat conversation** — map `instruction → system`, `input → user`, `output → assistant`, then render it with **Llama's chat template**. The notebook compares Mistral vs. Llama templates to show why the format is model-specific.
6. **Tokenize** the formatted text into `input_ids` / `attention_mask` and drop the raw text columns.
7. **Build a data collator** (`DataCollatorForLanguageModeling(mlm=False)`) for dynamic padding + automatic label creation, and set the pad token to the EOS token.
8. **Add LoRA adapters** — a dedicated, illustrated section explains *what LoRA is and why*, then `LoraConfig(r=64, ...)` wraps the model so only **~3.5%** of the parameters are trainable. The base model is deep-copied first so you can compare "before vs. after."
9. **Configure & run SFT training** with `trl`'s `SFTTrainer` for a short 60-step demo run (effective batch size 8, fp16, periodic eval). Watch the training/validation loss fall in the results table.
10. **Verify what changed** — the frozen int8 base weights are unchanged, while the small fp32 LoRA `A`/`B` matrices *were* updated (they even train in float32 while the base stays int8).
11. **Merge & push** — reload the model in full precision, attach the saved adapters, `merge_and_unload()` them into the weights, and push the standalone fine-tuned model + tokenizer to the Hub.
12. **Run inference** with both greedy and sampled (temperature/top-p/top-k + repetition penalty) decoding to watch the fine-tuned model turn English requests into Docker commands.

**Concepts covered:** [base vs. instruct models](#3-base-vs-instruct-models--chat-templates), chat templates & special tokens, [8-bit quantization](#1-quantization), [tokenizer usage vs. fine-tuning](#2-tokenization--and-why-the-tokenizer-is-not-fine-tuned), [LoRA / parameter-efficient fine-tuning](#5-lora-low-rank-adaptation), [supervised fine-tuning (SFT)](#4-fine-tuning-and-sft), dynamic padding & label masking, and adapter merging.

Every code cell is annotated with line-by-line comments, and each section has an explanatory markdown intro.

---

## Setup

### 1. Install dependencies

Each notebook installs what it needs in its first cell. To run locally instead of Colab:

```bash
pip install -U transformers bitsandbytes accelerate trl peft datasets huggingface_hub python-dotenv
```

> Note: `bitsandbytes` 8-bit/4-bit quantization requires a **CUDA GPU**.

### 2. Configure your Hugging Face token

Both notebooks read the token from a `.env` file (never commit this file). Copy the example and paste your token:

```bash
cp .env.example .env
```

Then edit `.env`:

```
HF_TOKEN=hf_your_token_here
```

Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), and make sure your account has accepted the [Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). In Google Colab, upload the `.env` file to the session's working directory.

### 3. Run

Open a notebook on a GPU runtime and run the cells top to bottom.

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU (required for the quantized model cells)
- A Hugging Face account with access to Llama 3.2
