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

Explains the difference between **base** and **instruct** models — the key being the **chat template** with special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) that instruct models are trained on — and then fine-tunes an instruct model end to end.

Starting from `meta-llama/Llama-3.2-1B-Instruct`, the notebook:
- Loads the model in 8-bit.
- Loads a **natural-language → Docker command** dataset.
- Formats each example with Llama's chat template.
- Tokenizes and batches the data (dynamic padding + label masking).
- Attaches **LoRA** adapters and runs **supervised fine-tuning (SFT)** with `trl`'s `SFTTrainer` — training only ~3.5% of the parameters.
- Merges the LoRA adapters into a full-precision model and pushes it to the Hub.
- Runs inference to translate English requests into Docker commands.

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
