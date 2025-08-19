# LLM Transformers Project - Medical Text Generation

This project demonstrates how to use Microsoft's Phi-4-mini-instruct model with the Hugging Face Transformers library for medical text generation tasks, specifically using Alzheimer's disease research abstracts.

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Authentication Setup

Instead of interactive login, this project uses a token-based approach:

1. Create a `.env` file (or use the provided `hf_token.env`) in your project root
2. Add your Hugging Face token:

```bash
# In hf_token.env
HUGGINGFACE_TOKEN=your_actual_token_here
```

3. Replace `your_actual_token_here` with your actual Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Run the Notebook

```bash
jupyter notebook phi4-transformers.ipynb
```

## Features

- **Raw Transformers API**: Direct model and tokenizer usage
- **Pipeline**: Simplified text generation pipeline
- **Training**: Complete training loop with medical datasets
- **Model Sharing**: Push trained models to Hugging Face Hub

## Models Used

- Microsoft Phi-4-mini-instruct for text generation
- Gaborandi/Alzheimer_pubmed_abstracts dataset for medical text training

## Dataset Information

The project uses the **Gaborandi/Alzheimer_pubmed_abstracts** dataset, which contains:
- PubMed abstracts related to Alzheimer's disease research
- High-quality, peer-reviewed medical literature
- Structured format suitable for text generation training
- Medical domain expertise for specialized applications

## Notes

- The notebook will automatically authenticate using your token from the environment file
- No more manual token pasting during execution
- Make sure to keep your token secure and never commit it to version control
- Medical datasets require careful handling and validation of generated content

## Environment (for issues)

Copy-and-paste the text below in your GitHub issue:

```text
- `Accelerate` version: 1.10.0
- Platform: Linux-6.8.0-1030-azure-x86_64-with-glibc2.35
- `accelerate` bash location: /mnt/batch/tasks/shared/LS_root/mounts/clusters/ucs-nc40-h100/code/phi4-ft/.venv/bin/accelerate
- Python version: 3.13.7
- Numpy version: 2.3.2
- PyTorch version: 2.8.0+cu128
- PyTorch accelerator: CUDA
- System RAM: 314.69 GB
- GPU type: NVIDIA H100 NVL
- `Accelerate` default config:
	- compute_environment: LOCAL_MACHINE
	- distributed_type: DEEPSPEED
	- mixed_precision: bf16
	- use_cpu: False
	- debug: False
	- num_processes: 1
	- machine_rank: 0
	- num_machines: 1
	- rdzv_backend: static
	- same_network: True
	- main_training_function: main
	- enable_cpu_affinity: False
	- deepspeed_config: {'gradient_accumulation_steps': 1, 'gradient_clipping': 1.0, 'offload_optimizer_device': 'none', 'offload_param_device': 'none', 'zero3_init_flag': True, 'zero3_save_16bit_model': True, 'zero_stage': 3}
	- downcast_bf16: no
	- tpu_use_cluster: False
	- tpu_use_sudo: False
	- tpu_env: []
```
