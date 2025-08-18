"""
Supervised Fine tuning phi4 on Medical Data
"""

import os
import logging
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
    logger.info("Logged in to Hugging Face Hub")
else:
    logger.warning("HUGGINGFACE_TOKEN not found in environment variables. Skipping login.")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

def apply_chat_template(data, tk):
    """
    Apply chat template to the dataset.
    """
    messages = data["messages"]
    data["text"] = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return data

train_data, test_data = load_dataset(
    "HuggingFaceH4/ultrachat_200k", split=["train_sft", "test_sft"]
    )

processed_train_dataset = train_data.map(
    apply_chat_template, fn_kwargs={"tk": tokenizer}, 
    remove_columns=train_data.column_names, num_proc=10
)
logger.info("Processed train dataset: %s samples", len(processed_train_dataset))

processed_test_dataset = test_data.map(
    apply_chat_template, fn_kwargs={"tk": tokenizer}, 
    remove_columns=test_data.column_names, num_proc=10
)
logger.info("Processed test dataset: %s samples", len(processed_test_dataset))

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct")