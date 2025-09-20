import torch
import unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Load structured dataset
dataset = load_dataset("json", data_files={"train": "ai_polished_reasons.jsonl"})[
    "train"
]


# Convert structured fields into a single text column with a fixed template
def format_data(data):
    return {
        "text": (
            f"### Situation:\n"
            f"Current Thrown Card: {data['current_thrown_card']}\n"
            f"Leading Card: {data['leading_card']}\n"
            f"Discarded Pile: {data['discarded_pile']}\n"
            f"Legal Cards: {data['legal_cards']}\n\n"
            f"### Output:\n"
            f"Best Card to Throw: {data['best_card_to_throw']}\n"
            f"Reason: {data['reason']}"
        )
    }


dataset = dataset.map(format_data)

# 2. Load your previously trained SFT Rules LoRA model
# This automatically loads base model + your LoRA weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./callbreak_agent/trained_model/callbreak_rules_lora",  # Path where you saved your rules LoRA
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    gpu_memory_utilization=0.9,
)

# If you saved only the LoRA adapter separately, merge manually like this:
# base_model, tokenizer = FastLanguageModel.from_pretrained("Qwen2.5-1.5B-Instruct", ...)
# base_model.load_lora("./rules_lora_adapter")  # load adapter weights

# 3. Apply / extend LoRA for Best Card SFT
# We re-use the same LoRA modules so that training continues seamlessly
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./best-card-lora",  # new output directory
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),
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

# 6. Train
trainer.train()

# 7. Save LoRA adapter
model.save_pretrained(
    "./callbreak_agent/trained_model/best_card_lora", save_adapter=True
)  # adapter only
tokenizer.save_pretrained("./callbreak_agent/trained_model/best_card_lora")

print("✅ Best Card LoRA training completed.")
