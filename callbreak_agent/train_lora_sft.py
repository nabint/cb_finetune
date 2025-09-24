import os
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from unsloth import FastLanguageModel  # keeps parity with your environment

DATA_FILE = "ai_polished_reasons.jsonl"   # your dataset
OUTPUT_DIR = "./lora_sft_best"
EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 1               # increase with more GPU memory and use gradient_accumulation_steps
GRAD_ACCUM = 16              # effective batch = BATCH_SIZE * GRAD_ACCUM
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset and create simple card-only prompts
raw = load_dataset("json", data_files={"train": DATA_FILE})["train"]

def make_prompt_card_only(ex):
    prompt = (
        "### ROLE: You are a Callbreak expert.\n"
        "Decide the single best card to throw.\n\n"
        f"Discarded Pile: {ex.get('discarded_pile', [])}\n"
        f"Current Round Cards Played: {ex.get('current_thrown_card', [])}\n"
        f"Leading Card: {ex.get('leading_card','')}\n"
        f"Legal Cards: {ex.get('legal_cards', [])}\n\n"
        "### STRICT OUTPUT FORMAT (one line):\nCard:"
    )
    # prefer keys matching your dataset
    card = ex.get("best_card_to_throw") or ex.get("best_card") or ""
    return {"prompt": prompt, "card": card}

mapped = raw.map(make_prompt_card_only)

# Load model & tokenizer (match your environment)
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
base_model.config.use_cache = True
base_model = prepare_model_for_kbit_training(base_model)

# Ensure pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

# Wrap with LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],  # common projection names; adjust if needed
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Tokenize such that only the card tokens contribute to the loss
def tokenize_for_card_only(ex):
    prompt = ex["prompt"]
    card = ex["card"].strip()
    full = prompt + " " + card
    # encode without adding special tokens to control label positions
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    input_ids = full_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)
    # label only the card token positions (suffix)
    start = len(prompt_ids)
    for i in range(start, len(input_ids)):
        labels[i] = input_ids[i]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

tokenized = mapped.map(tokenize_for_card_only, remove_columns=mapped.column_names)

# TrainingArguments + Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="epoch",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# Train
trainer.train()
# Save only the LoRA adapter (small)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved LoRA to:", OUTPUT_DIR)
