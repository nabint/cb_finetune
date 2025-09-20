import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer


# 1. Load dataset
dataset = load_dataset("json", data_files={"train": "callbreak_rules.jsonl"})["train"]

# Convert to text-to-text format (chat-like)
def format_example(example):
    prompt = example["prompt"]
    answer = example["response"]
    return {
        "text": f"### Question:\n{prompt}\n\n### Answer:\n{answer}"
    }

dataset = dataset.map(format_example)

# 2. Load base model + tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen2.5-1.5B-Instruct", 
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,                    # save VRAM
    gpu_memory_utilization=0.9,
)

# 3. Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./callbreak_agent/callbreak-rules-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),       # use bf16 if GPU supports
    remove_unused_columns=False,
)

# 5. SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()


# 6. Save LoRA adapter
model.save_pretrained("./callbreak_agent/trained_model/best_card_lora", save_adapter=True)  # adapter only
tokenizer.save_pretrained("./callbreak_agent/trained_model/best_card_lora")

print("✅ SFT training completed.")