# LLM-Finetuning

A hands-on, notebook-based walkthrough of core techniques for working with and fine-tuning large language models, using Llama 3.2 and the Hugging Face ecosystem (`transformers`, `bitsandbytes`, `peft`, `trl`).

Every notebook is designed to run on a **GPU runtime** (e.g. Google Colab) and authenticates to the Hugging Face Hub with a token read from a local `.env` file.

## Notebooks

### 1. [`1-Quantization.ipynb`](1-Quantization.ipynb) — Model quantization

Explores **quantization**: reducing the numerical precision used to store a model's weights to shrink its memory footprint (and enable running large models on modest GPUs) with minimal quality loss.

The same model, `meta-llama/Llama-3.2-1B-Instruct`, is loaded at four precisions and compared:

| Precision | dtype | Bytes / weight | Approx. size |
|-----------|-----------|----------------|--------------|
| 32-bit | `float32` | 4 | ~4.6 GB |
| 16-bit | `bfloat16` | 2 | ~2.3 GB |
| 8-bit | `int8` | ~1 | ~1.4 GB |
| 4-bit | `NF4` | ~0.5 | ~0.9 GB |

You will see how to:
- Load a model in 4-, 8-, 16-, and 32-bit with `BitsAndBytesConfig`.
- Measure and compare each variant's memory footprint.
- Inspect how the raw weights change (floating-point vs. integer) after quantization.
- See how the layer types differ per precision (`Linear`, `Linear8bitLt`, `Linear4bit`).
- Measure real GPU memory usage.

### 2. [`2-InstructVsBaseModel.ipynb`](2-InstructVsBaseModel.ipynb) — Instruct vs. base models & LoRA fine-tuning

A complete, end-to-end fine-tuning walkthrough built around one core idea: the difference between a **base** model and an **instruct** model.

- A **base** model is only pretrained to predict the next token on raw text.
- An **instruct** model is further trained (SFT / RLHF) to follow instructions, and it expects inputs formatted with a **chat template** — role-based `system` / `user` / `assistant` turns wrapped in model-specific special tokens (for Llama 3.2: `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`). The notebook renders the *same* conversation with both the Mistral and Llama templates to make the contrast concrete.

Using `meta-llama/Llama-3.2-1B-Instruct`, it then fine-tunes the model to translate plain-English requests into **Docker commands**, covering the full workflow:

1. **Load** the instruct model in 8-bit (`BitsAndBytesConfig`) so it fits on a small GPU.
2. **Load & split** the [`MattCoddity/dockerNLcommands`](https://huggingface.co/datasets/MattCoddity/dockerNLcommands) dataset (80/20 train/validation).
3. **Format** each row into a `system` / `user` / `assistant` conversation and render it with Llama's chat template.
4. **Tokenize** the data and batch it with `DataCollatorForLanguageModeling` (dynamic padding + label masking, `-100` on pad positions).
5. **Add LoRA adapters** — a dedicated, illustrated section explains *what LoRA is and why we use it*, then `LoraConfig` (rank 64) makes only **~3.5%** of the parameters trainable.
6. **Train** with `trl`'s `SFTTrainer` for a short 60-step demo run.
7. **Inspect** how the frozen int8 base weights stay unchanged while the fp32 LoRA matrices are updated.
8. **Merge** the adapters into a full-precision model (`merge_and_unload`) and **push** it to the Hub.
9. **Run inference** with both greedy and sampled decoding to watch the fine-tuned model produce Docker commands.

**Concepts covered:** base vs. instruct models, chat templates & special tokens, 8-bit quantization, LoRA / parameter-efficient fine-tuning, supervised fine-tuning (SFT), dynamic padding & label masking, and adapter merging.

Every code cell is annotated with line-by-line comments, and each section has an explanatory markdown intro.

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

## Requirements

- Python 3.10+
- A CUDA-capable GPU (required for the quantized model cells)
- A Hugging Face account with access to Llama 3.2
