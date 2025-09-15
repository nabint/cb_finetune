import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    dataset = load_dataset("json", data_files={"train": "data.jsonl"})["train"]
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Load model and tokenizer with Unsloth
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="qwen2.5-0.5b-instruct",
        max_seq_length=4096,  # Increased for longer explanations
        dtype=None,
        load_in_4bit=True,
    )
    print("Model and tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Enhanced LoRA configuration
try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # Higher rank for better domain adaptation
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth"
    )
    print("LoRA applied to model")
except Exception as e:
    print(f"Error applying LoRA: {e}")
    exit(1)

# Better formatting with system prompt
def formatting_prompts_func(examples):
    texts = []
    system_prompt = "You are a callbreak card game expert. Provide accurate information about callbreak rules, strategies, and gameplay."
    
    for prompt, response in zip(examples["prompt"], examples["response"]):
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        texts.append(text)
    return {"text": texts}

# Apply formatting to datasets
train_dataset = dataset["train"].map(formatting_prompts_func, batched=True)
eval_dataset = dataset["test"].map(formatting_prompts_func, batched=True)

# Optimized training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,  # More epochs for domain specialization
    learning_rate=5e-5,  # Lower learning rate
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    warmup_steps=100,
    max_steps=-1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
)

# Create SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,  # Better efficiency
    args=training_args,
)

# Start training
try:
    print("Starting training...")
    trainer.train()
    print("Training completed successfully")
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save the model
try:
    model.save_pretrained("./outputs/final_model")
    tokenizer.save_pretrained("./outputs/final_model")
    print("✅ Training complete. Model saved to ./outputs/final_model")
except Exception as e:
    print(f"Error saving model: {e}")

print("Fine-tuning process finished!")