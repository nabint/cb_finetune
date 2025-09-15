# main.py

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# 1. Load dataset
# Load JSONL dataset into HuggingFace Datasets
dataset = load_dataset("json", data_files={"train": "data.jsonl"})["train"]

# Split into train (90%) and validation (10%)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ---------------------------
# 2. Load base model with Unsloth QLoRA
# ---------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # change if you want another model
    max_seq_length=2048,
    dtype=None, # auto-detects float16/bfloat16
    load_in_4bit=True,
)

# Enable gradient checkpointing + LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",  # applies LoRA to all linear layers
    use_gradient_checkpointing="unsloth",  # memory efficient
    random_state=42,
    use_rslora=True,  # rank-stabilized LoRA
    loftq_config=None,
)

# ---------------------------
# 3. Training configuration
# ---------------------------
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",  # set to "wandb" or "tensorboard" if you want logging
)

# ---------------------------
# 4. Trainer
# ---------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="prompt",  # 👈 change if your jsonl uses another key
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# ---------------------------
# 5. Train
# ---------------------------
trainer.train()

# ---------------------------
# 6. Save model + tokenizer
# ---------------------------
model.save_pretrained("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")
