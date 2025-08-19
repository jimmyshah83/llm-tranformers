"""
Supervised Fine tuning phi4 on Medical Data
"""

import os
import logging
import sys
from dotenv import load_dotenv
from huggingface_hub import login
import datasets
from datasets import load_dataset
import transformers
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

load_dotenv(override=True)

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

def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

train_dataset, test_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=["train_sft", "test_sft"])

processed_train_dataset = train_dataset.map(
	apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, 
	remove_columns=train_dataset.column_names, num_proc=10
)
logger.info("Processed train dataset: %s samples", len(processed_train_dataset))

processed_test_dataset = test_dataset.map(
	apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, 
	remove_columns=test_dataset.column_names, num_proc=10
)
logger.info("Processed test dataset: %s samples", len(processed_test_dataset))

training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 1,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": True},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    }

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    train_conf.local_rank,
    train_conf.device,
    train_conf.n_gpu,
    bool(train_conf.local_rank != -1),
    train_conf.fp16,
)
logger.info("Training/evaluation parameters %s", train_conf)
logger.info("PEFT parameters %s", peft_conf)

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", attn_implementation="flash_attention_2", use_cache=False)

# Training
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluation
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

trainer.save_model(train_conf.output_dir)

# Push model to Hugging Face Hub
# trainer.push_to_hub(
#     repo_id="jimmyshah83/phi4-mini-instruct-finetuned",
#     commit_message="Fine-tuned Phi-4 mini on OpenMathInstruct-2 dataset"
# )

# Also push the tokenizer
# tokenizer.push_to_hub(
#     repo_id="jimmyshah83/phi4-mini-instruct-finetuned",
#     commit_message="Fine-tuned tokenizer for Phi-4 mini"
# )
