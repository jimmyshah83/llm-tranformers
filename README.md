# LLM Transformers Project

This project demonstrates how to use Microsoft's Phi-4-mini-instruct model with the Hugging Face Transformers library.

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
- **Training**: Complete training loop with datasets
- **Model Sharing**: Push trained models to Hugging Face Hub

## Models Used

- Microsoft Phi-4-mini-instruct for text generation
- Rotten Tomatoes dataset for training examples

## Notes

- The notebook will automatically authenticate using your token from the environment file
- No more manual token pasting during execution
- Make sure to keep your token secure and never commit it to version control
