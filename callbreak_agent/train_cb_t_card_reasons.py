"""
train_callbreak_with_reasons.py

- SFT then RLHF (GRPO) pipeline for Callbreak card + reason generation.
- Expects dataset rows that include:
    discarded_pile, current_thrown_card, leading_card, legal_cards, best_card_to_throw, reason

Install (example):
pip install transformers datasets accelerate bitsandbytes trl peft

Adjust hyperparams below as needed.
"""

import os
import re
import math
import json
from typing import List, Optional, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import GRPOTrainer, GRPOConfig

# Optional: for 4-bit training/peft if you want later
from peft import prepare_model_for_kbit_training, PeftModel

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "data/callbreak_dataset.jsonl"  # one JSON object per line
MODEL_NAME = "gpt2"  # change to your base model, e.g., "mistral-7b" or equivalent
OUTPUT_DIR = "./callbreak_model"
SFT_CHECKPOINT = os.path.join(OUTPUT_DIR, "sft")
RLHF_CHECKPOINT = os.path.join(OUTPUT_DIR, "rlhf")

SFT_BATCH_SIZE = 4
SFT_EPOCHS = 3
SFT_LR = 2e-5

RL_BATCH_SIZE = 2
RL_EPOCHS = 2000  # number of optimization steps for RL loop (not epochs)
RL_LR = 2e-5
MAX_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------------------------
# Utilities (replace with your project's versions if you have them)
# ---------------------------
CARD_RE = re.compile(r"(?:Ace|King|Queen|Jack|10|9|8|7|6|5|4|3|2) of (?:spades|hearts|diamonds|clubs)", re.I)


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_texts(texts: List[str]) -> List[str]:
    # Basic normalizer: strip, collapse multiple spaces, unify newlines
    out = []
    for t in texts:
        if t is None:
            out.append("")
            continue
        s = t.strip()
        s = re.sub(r"\s+", " ", s)
        out.append(s)
    return out


def extract_card(text: str) -> Optional[str]:
    """
    Attempt to extract the first card mention in the text.
    Looks for patterns like 'Card: Ace of Spades' or plain 'Ace of Spades'.
    """
    if not text:
        return None
    text_low = text.lower()
    # Try to find 'card:' label first
    m = re.search(r"card\s*:\s*([A-Za-z0-9 ]+ of [A-Za-z]+)", text, re.I)
    if m:
        return m.group(1).strip()
    # fallback to regex
    m2 = CARD_RE.search(text)
    if m2:
        return m2.group(0).strip()
    return None


def broadcast(value, n):
    if value is None:
        return [None] * n
    if isinstance(value, list):
        if len(value) == n:
            return value
        # broadcast single-item list
        if len(value) == 1:
            return value * n
        # otherwise fallback
        return [value[0]] * n
    return [value] * n


# ---------------------------
# Format function for dataset rows
# ---------------------------
def format_example_for_model(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create prompt and target completion for supervised fine-tuning.
    """
    prompt = f"""
### ROLE: You are a Callbreak expert AI.
### LANGUAGE: Answer strictly in ENGLISH only.
### GOAL: Decide the single best card to throw from the Legal Cards list and explain why.

### CALLBREAK REFERENCE:
- Callbreak is a 4-player trick-taking game.
- The trump suit is ALWAYS Spades.
- Players must follow the leading suit if possible. Otherwise play Spade or any other card.
- Highest card of the leading suit wins unless a Spade (trump) is played. Among Spades, the highest Spade wins.
- Lower cards are often used to save higher cards for later.
- Cards already discarded help infer which cards remain in play.

### GIVEN STATE:
- Discarded Pile: {example.get("discarded_pile", [])}
- Current Round Cards Played: {example.get("current_thrown_card", [])}
- Leading Card: {example.get("leading_card", "")}
- Legal Cards: {example.get("legal_cards", [])}

### STRICT OUTPUT FORMAT:
Card: <exact best card>
Reason: <plausible explanation>
"""

    completion = f"Card: {example.get('best_card_to_throw','')}\nReason: {example.get('reason','')}"
    return {"prompt": prompt, "completion": completion, "legal_cards": example.get("legal_cards", [])}


# ---------------------------
# Reward function for RLHF (GRPO)
# ---------------------------
def reward_function(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    discarded_pile=None,
    current_thrown_card=None,
    leading_card=None,
    best_card_to_throw=None,
    legal_cards=None,
    reason=None,
    trainer_state=None,
):
    """
    Reward both a correct card and a plausible reason.
    This is heuristic — you can replace with an LLM-judge later.
    """
    norm_completions = normalize_texts(completions)

    true_cards = broadcast(best_card_to_throw, len(norm_completions))
    legal_cards_b = broadcast(legal_cards, len(norm_completions))

    rewards = []
    for comp_text, true_card, legal_list in zip(norm_completions, true_cards, legal_cards_b):
        reward = 0.0
        pred_card = extract_card(comp_text)

        legal_set = set(x.lower() for x in (legal_list or []))

        # --- Card scoring
        if not pred_card:
            reward -= 30.0  # missing card
        else:
            pc_low = pred_card.lower().strip()
            if legal_set and pc_low not in legal_set:
                reward -= 50.0  # illegal card
            if true_card:
                tc_low = str(true_card).lower().strip()
                if pc_low == tc_low:
                    reward += 12.0
                else:
                    reward -= 25.0

        # --- Formatting penalties
        if not comp_text.strip().lower().startswith("card:"):
            reward -= 10.0
        if "\n" in comp_text.strip():
            # we expect Card + Reason separated by newline; but penalize too many newlines
            newline_count = comp_text.count("\n")
            if newline_count > 3:
                reward -= 6.0

        if "```" in comp_text or "print(" in comp_text:
            reward -= 20.0

        # --- Reason scoring (heuristic)
        reason_score = 0.0
        reason_text = ""
        low_text = comp_text.lower()
        if "reason:" in low_text:
            reason_text = comp_text.split("reason:", 1)[-1].strip()
        else:
            # try to find some sentence after card
            parts = comp_text.split("\n")
            if len(parts) >= 2:
                reason_text = parts[1].strip()

        if reason_text:
            words = reason_text.split()
            if len(words) < 4:
                reason_score -= 5.0  # too short, likely uninformative
            else:
                reason_score += 4.0  # rewarded for giving a non-trivial reason

            # heuristic checks: mention of relevant tokens improves plausibility
            tokens = {"spade", "trump", "leading", "follow", "discard", "safe", "high", "low", "void", "break"}
            matches = sum(1 for t in tokens if t in reason_text.lower())
            reason_score += min(matches, 3) * 1.5
        else:
            reason_score -= 8.0  # no reason provided

        reward += reason_score

        rewards.append(float(reward))

    return rewards


# ---------------------------
# Data loading & preparation
# ---------------------------
def load_jsonl_to_dataset(path: str) -> Dataset:
    """Load JSONL into Hugging Face Dataset and format it."""
    raw = load_dataset("json", data_files=path, split="train")
    # Map the format function
    def _format(example):
        out = format_example_for_model(example)
        # store fields as strings for tokenization
        return out

    ds = raw.map(_format)
    return ds


# ---------------------------
# Tokenization helpers
# ---------------------------
def build_tokenized_dataset(ds: Dataset, tokenizer: AutoTokenizer, max_length=MAX_LENGTH):
    """
    Create input_ids / labels for causal LM using prompt+completion concatenation.
    We'll use the common pattern: input = prompt + completion, labels mask prompt tokens to -100
    """
    def tokenize_fn(ex):
        prompt = ex["prompt"]
        completion = ex["completion"]
        full = (prompt + completion).strip()
        tokenized = tokenizer(
            full,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # find prompt token length so we can mask them in labels
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
        prompt_len = len(prompt_tokens["input_ids"])

        labels = tokenized["input_ids"].copy()
        # mask prompt part with -100
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        tokenized["labels"] = labels
        return tokenized

    tok_ds = ds.map(tokenize_fn, remove_columns=ds.column_names, batched=False)
    return tok_ds


# ---------------------------
# SFT training
# ---------------------------
def run_sft(training_dataset: Dataset):
    print("Loading tokenizer and model for SFT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Ensure tokenizer has BOS/EOS if needed by model
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    print("Tokenizing dataset for SFT...")
    tok_ds = build_tokenized_dataset(training_dataset, tokenizer, max_length=MAX_LENGTH)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=SFT_CHECKPOINT,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        learning_rate=SFT_LR,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        data_collator=data_collator,
    )

    print("Starting SFT...")
    trainer.train()
    print("Saving SFT model to", SFT_CHECKPOINT)
    trainer.save_model(SFT_CHECKPOINT)

    return SFT_CHECKPOINT, tokenizer, model


# ---------------------------
# RLHF (GRPO) training
# ---------------------------
def run_rlhf(sft_checkpoint: str, tokenizer: AutoTokenizer, base_model: AutoModelForCausalLM, ds_for_rl: Dataset):
    # Load model from sft checkpoint for RL stage
    print("Preparing model for RLHF...")
    model = AutoModelForCausalLM.from_pretrained(sft_checkpoint).to(DEVICE)

    # Optionally prepare for k-bit training / PEFT if needed:
    # model = prepare_model_for_kbit_training(model)

    # GRPOConfig — tune to taste
    config = GRPOConfig(
        lr=RL_LR,
        batch_size=RL_BATCH_SIZE,
        ppo_epochs=1,
        pfactor=0.95,  # placeholder; tune for your setting
        # other config params can be added
    )

    # For RL, we provide prompts and have the trainer generate continuations.
    # Build a "prompt only" dataset by using the prompt fields only and letting
    # the RL trainer sample completions.
    prompt_ds = ds_for_rl.map(lambda ex: {"text": ex["prompt"]})

    # GRPOTrainer expects a model that is a huggingface transformers model
    trn = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=prompt_ds,  # dataset of prompts
        reward_fn=reward_function,
        output_dir=RLHF_CHECKPOINT,
        config=config,
        # additional arguments (e.g. generation kwargs) can be passed here
    )

    print("Starting RLHF (GRPO) tuning...")
    trn.train()
    print("Saving RLHF model to", RLHF_CHECKPOINT)
    trn.save_model(RLHF_CHECKPOINT)


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    set_seed()

    print("Loading dataset...")
    ds = load_jsonl_to_dataset(DATA_PATH)

    # Split dataset for SFT (train) and RL (we'll use same examples' prompts)
    ds = ds.shuffle(seed=SEED)
    train_test = ds.train_test_split(test_size=0.05, seed=SEED)
    sft_train_ds = train_test["train"]
    rl_ds = train_test["test"]  # use small set for RL prompts; or reuse train prompts

    # Run SFT
    sft_ckpt, tokenizer, sft_model = run_sft(sft_train_ds)

    # For RL, we need tokenizer and base model
    # Use sft checkpoint as base model for RL
    run_rlhf(sft_ckpt, tokenizer, sft_model, rl_ds)

    print("All done.")


if __name__ == "__main__":
    main()
