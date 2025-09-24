import re
import torch
import random
import traceback
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel, prepare_model_for_kbit_training

# Utilities

CARD_RE = re.compile(
    r"(?:Ace|King|Queen|Jack|10|9|8|7|6|5|4|3|2) of (?:Clubs|Diamonds|Hearts|Spades)",
    re.IGNORECASE,
)


def extract_card(text: str):
    if not text:
        return None
    m = CARD_RE.search(text)
    return m.group(0) if m else None


def normalize_texts(values, tokenizer):
    """Normalize completions to list of text strings."""
    if isinstance(values, torch.Tensor):
        return [
            tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in values
        ]
    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            if isinstance(v, torch.Tensor):
                out.append(tokenizer.decode(v.tolist(), skip_special_tokens=True))
            elif isinstance(v, list) and v and isinstance(v[0], int):
                out.append(tokenizer.decode(v, skip_special_tokens=True))
            else:
                out.append(str(v))
        return out
    return [str(values)]


def broadcast(value, length):
    """Broadcast single value to a list of length."""
    return value if isinstance(value, (list, tuple)) else [value] * length


# Load Data

print("Loading dataset...")
raw = load_dataset("json", data_files={"train": "ai_polished_reasons_4.jsonl"})["train"]


def format_data(example):
    return {
        "prompt": f"""
        ### ROLE: You are a Callbreak expert AI.
        ### LANGUAGE: Answer strictly in ENGLISH only.
        ### GOAL: Decide the single best card to throw from the Legal Cards list, following Callbreak rules. 
        Only output the card as specified.

        ### CALLBREAK REFERENCE (rules to guide your choice):
        - Callbreak is a 4-player trick-taking game.
        - The trump suit is ALWAYS Spades. Spades beat all other suits unless a higher Spade is played.
        - Players must follow the leading suit if possible. If no card of the leading suit is available, they may throw a Spade (trump) or any other card.
        - Highest card of the leading suit wins unless a Spade (trump) is played. Among Spades, the highest Spade wins.
        - Lower cards are often used to save higher cards for later, unless a win is critical.
        - Cards already discarded help you infer which cards remain in play.

        ### GIVEN STATE:
        - Discarded Pile (all cards played so far): {example["discarded_pile"]}
        - Current Round Cards Played: {example["current_thrown_card"]}
        - Leading Card of this trick: {example["leading_card"]}
        - Your Legal Cards (cards you are allowed to play): {example["legal_cards"]}

        ### STRICT OUTPUT FORMAT:
        Card: <exact best card to throw from the Legal Cards above>

        ### IMPORTANT:
        - Use the discarded pile to estimate which cards remain.
        - Always consider Spades as trump.
        - Never output commentary, code, or multiple cards.
        - Only output one card in the specified format.
        """,
        "best_card_to_throw": example.get("best_card_to_throw", ""),
        "reason": example.get("reason", ""),  # keep reasoning in dataset but not used for reward
        "legal_cards": example.get("legal_cards", []),
    }

dataset = raw.map(format_data)
print(f"Dataset size: {len(dataset)}")

# Load Model

print("Loading model...")
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

base_model.config.use_cache = True
base_model = prepare_model_for_kbit_training(base_model)

# Load LoRA adapter
lora_path = "./callbreak_agent/trained_model/callbreak_rl_grpo_3"
try:
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
    print(f"Loaded LoRA from: {lora_path}")
except Exception as e:
    print(f"Warning: Could not load LoRA: {e}")
    model = base_model

# Make LoRA trainable only
trainable_params = 0
for name, param in model.named_parameters():
    if any(x in name.lower() for x in ["lora", "ada", "adapter"]):
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False
print(f"Trainable parameters: {trainable_params}")

# Tokenizer & model config fixes
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = True

# Safe generate wrapper
original_generate = model.generate


def safe_generate(*args, **kwargs):
    try:
        return original_generate(*args, **kwargs)
    except AttributeError as e:
        msg = str(e).lower()
        if any(k in msg for k in ["shape", "past_key_values", "nonetype"]):
            kwargs.setdefault("use_cache", False)
            kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
            return original_generate(*args, **kwargs)
        raise


model.generate = safe_generate

# Reward Function


# --- reward_function (replace your current reward_function) ---
def reward_function(
    prompts,
    completions,
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
    Strongly penalize outputs that:
      - contain a card not in legal_cards (huge negative)
      - are different from best_card_to_throw (big negative)
    We ignore reason matching (no positive reward from reason).
    """
    norm_completions = normalize_texts(completions, tokenizer)

    # broadcast arrays
    true_cards = broadcast(best_card_to_throw, len(norm_completions))
    legal_cards_b = broadcast(legal_cards, len(norm_completions))

    rewards = []
    for i, (comp_text, true_card, legal_list) in enumerate(
        zip(norm_completions, true_cards, legal_cards_b)
    ):
        reward = 0.0
        comp_text_l = comp_text.lower()
        pred_card = extract_card(comp_text)

        # Normalize legal_cards to lowercase strings
        legal_set = set()
        if legal_list:
            # legal_list might be a string representation, list, or other; try flexible handling
            if isinstance(legal_list, str):
                # attempt to parse simple Python-like list string
                try:
                    # if it's like "['A of Hearts', ...]" eval is risky but dataset trusted; safer fallback:
                    import ast

                    parsed = ast.literal_eval(legal_list)
                    if isinstance(parsed, (list, tuple)):
                        legal_set = {str(x).lower() for x in parsed}
                except Exception:
                    # fallback: split on commas
                    legal_set = {
                        x.strip().lower() for x in legal_list.split(",") if x.strip()
                    }
            elif isinstance(legal_list, (list, tuple)):
                legal_set = {str(x).lower() for x in legal_list}
            else:
                legal_set = {str(legal_list).lower()}

        if not pred_card:
            reward -= 30.0
            # still continue other checks (format penalties)
        else:
            pc_low = pred_card.lower().strip()
            if legal_set and pc_low not in legal_set:
                reward -= 50.0

            pred_suit = (
                pred_card.split(" of ")[-1].lower() if " of " in pred_card else ""
            )
            true_suit = (
                true_card.split(" of ")[-1].lower()
                if true_card and " of " in true_card
                else ""
            )

            if true_card:
                tc_low = str(true_card).lower().strip()
                if pc_low == tc_low:
                    reward += 12.0

                elif pred_suit != true_suit and true_suit == "spades":
                    reward -= 30
                else:
                    # mismatch -> strong negative (we want the model to learn only the dataset card)
                    reward -= 25.0

        if not comp_text.strip().lower().startswith("card:"):
            reward -= 10.0

        if "\n" in comp_text.strip():
            # allow if they accidentally appended whitespace, but heavy for extra content
            reward -= 8.0

        if "```" in comp_text or "print(" in comp_text:
            reward -= 20.0

        word_count = len(comp_text_l.split())
        if word_count > 35:
            reward -= 4.0
        elif word_count <= 1:
            reward -= 6.0
        else:
            reward += 0.5  # slight positive for concise answers

        if any("\u4e00" <= ch <= "\u9fff" for ch in comp_text):
            reward -= 5.0

        rewards.append(float(reward))

    return rewards


# Trainer

grpo_config = GRPOConfig(
    learning_rate=5e-5,
    num_generations=4,
    max_steps=100,
    loss_type="grpo",
    epsilon=0.15,
    per_device_train_batch_size=4,
)

print("Creating trainer...")
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_funcs=[reward_function],
    gen_kwargs={
        "max_new_tokens": 64,
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    },
)

# Test Generation

print("Testing generation...")
try:
    test_prompt = dataset[0]["prompt"]
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = (
        model.device
        if hasattr(model, "device")
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"Test generation: {generated[:300]}...")
except Exception as e:
    print(f"Generation test failed: {e}")

#  Training

print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    traceback.print_exc()

# Save Model

print("Saving model...")
try:
    save_path = "./callbreak_agent/trained_model/callbreak_rl_grpo_4"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")
except Exception as e:
    print(f"Save failed: {e}")

print("Script completed.")
