# Medical Dataset Example for Phi Training
# Add this to your notebook or run as a separate script

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

def load_medical_dataset():
    """Load and explore the MedQA dataset"""
    print("Loading MedQA dataset...")
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_source")
    print(f"Dataset loaded: {dataset}")
    
    # Show sample data
    print("\nSample data from train set:")
    sample = dataset["train"][0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Subject: {sample['subject_name']}")
    
    return dataset

def format_for_phi_training(dataset):
    """Format the dataset for Phi instruction tuning"""
    def format_example(example):
        # Create instruction-response format
        instruction = f"Answer this medical question: {example['question']}"
        response = example['answer']
        
        # Combine into a single text for causal LM training
        full_text = f"Instruction: {instruction}\nResponse: {response}\n\n"
        
        return {
            "text": full_text,
            "instruction": instruction,
            "response": response,
            "subject": example['subject_name']
        }
    
    print("Formatting dataset for Phi training...")
    formatted_dataset = dataset.map(format_example)
    return formatted_dataset

def setup_phi_model():
    """Load Phi model and tokenizer"""
    print("Loading Phi-4 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def tokenize_medical_data(dataset, tokenizer):
    """Tokenize the medical dataset"""
    def tokenize_function(examples):
        # Tokenize the full text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,  # Adjust based on your needs
            return_tensors="pt"
        )
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def create_training_config():
    """Create training arguments for medical dataset"""
    training_args = TrainingArguments(
        output_dir="phi4-medical-qa",
        learning_rate=1e-4,
        per_device_train_batch_size=4,  # Reduced for medical text complexity
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=True,
        report_to="none",  # Disable wandb/tensorboard if not needed
    )
    return training_args

def main():
    """Main function to demonstrate medical dataset usage"""
    print("=== Medical Dataset Training Example for Phi ===\n")
    
    # 1. Load dataset
    dataset = load_medical_dataset()
    
    # 2. Format for training
    formatted_dataset = format_for_phi_training(dataset)
    
    # 3. Load model and tokenizer
    model, tokenizer = setup_phi_model()
    
    # 4. Tokenize data
    tokenized_dataset = tokenize_medical_data(formatted_dataset, tokenizer)
    
    # 5. Create training config
    training_args = create_training_config()
    
    # 6. Setup trainer (commented out to avoid actual training)
    print("\n=== Training Setup Complete ===")
    print("To start training, uncomment the following lines:")
    print("""
    from transformers import DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Use causal language modeling
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    """)
    
    # 7. Show dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Training samples: {len(tokenized_dataset['train'])}")
    print(f"Validation samples: {len(tokenized_dataset['validation'])}")
    print(f"Test samples: {len(tokenized_dataset['test'])}")
    
    # 8. Show sample tokenized data
    print(f"\n=== Sample Tokenized Data ===")
    sample = tokenized_dataset["train"][0]
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Attention mask length: {len(sample['attention_mask'])}")
    
    # Decode sample
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"Decoded text: {decoded_text[:200]}...")

if __name__ == "__main__":
    main()
