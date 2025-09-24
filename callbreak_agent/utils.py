import torch
import re

from unsloth import FastLanguageModel
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset


def broadcast(value, length):
    return value if isinstance(value, (list, tuple)) else [value] * length


def extract_card(text: str):
    CARD_RE = re.compile(
        r"(?:Ace|King|Queen|Jack|10|9|8|7|6|5|4|3|2) of (?:Clubs|Diamonds|Hearts|Spades)",
        re.IGNORECASE,
    )

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


# Data Helper Functions
def format_data(example):
    """Format dataset rows into prompt/label format."""
    return {
        "prompt": f"""
            ### ROLE: You are a Callbreak expert AI.
            ### LANGUAGE: Answer strictly in ENGLISH only.
            ### GOAL: Decide the single best card to throw from the Legal Cards list, following Callbreak rules. 
            Only output the card as specified.

            ### CALLBREAK REFERENCE (rules to guide your choice):
            - Callbreak is a 4-player trick-taking game.
            - The trump suit is ALWAYS Spades.
            - Players must follow the leading suit if possible. Otherwise play Spade or any other card.
            - Highest card of the leading suit wins unless a Spade (trump) is played. Among Spades, the highest Spade wins.
            - Lower cards are often used to save higher cards for later.
            - Cards already discarded help infer which cards remain in play.

            ### GIVEN STATE:
            - Discarded Pile: {example["discarded_pile"]}
            - Current Round Cards Played: {example["current_thrown_card"]}
            - Leading Card: {example["leading_card"]}
            - Legal Cards: {example["legal_cards"]}

            ### STRICT OUTPUT FORMAT:
            Card: <exact best card to throw from the Legal Cards above>

            ### IMPORTANT:
            - Use the discarded pile to estimate which cards remain.
            - Always consider Spades as trump.
            - Never output commentary, code, or multiple cards.
            - Only output one card in the specified format.
        """,
        "best_card_to_throw": example.get("best_card_to_throw", ""),
        "reason": example.get("reason", ""),
        "legal_cards": example.get("legal_cards", []),
    }


def load_and_prepare_dataset(file_path):
    """Load and format dataset."""
    raw = load_dataset("json", data_files={"train": file_path})["train"]
    return raw.map(format_data)


# Model Helper Functions
def load_model_and_tokenizer(lora_path=None):
    """Load base model and optional LoRA adapter."""
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False, # Turn on this to save VRAM
        use_cache=True,
        use_gradient_checkpointing=False,
    )

    base_model = prepare_model_for_kbit_training(base_model) # No need to call this if no quantization has been applied

    # Try loading LoRA adapter
    model = base_model
    if lora_path:
        try:
            model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
            print(f"Loaded LoRA from: {lora_path}")
        except Exception as e:
            print(f"Warning: Could not load LoRA: {e}")

    # Make LoRA trainable only
    trainable_params = 0
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ["lora", "ada", "adapter"]):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(f"Trainable parameters: {trainable_params}")

    # Tokenizer fixes
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# Reward Function for GRPO
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
    norm_completions = normalize_texts(completions)

    true_cards = broadcast(best_card_to_throw, len(norm_completions))
    legal_cards_b = broadcast(legal_cards, len(norm_completions))

    rewards = []
    for comp_text, true_card, legal_list in zip(
        norm_completions, true_cards, legal_cards_b
    ):
        reward = 0.0
        pred_card = extract_card(comp_text)

        legal_set = set(str(x).lower() for x in (legal_list or []))

        if not pred_card:
            reward -= 30.0
        else:
            pc_low = pred_card.lower().strip()
            if legal_set and pc_low not in legal_set:
                reward -= 50.0

            if true_card:
                tc_low = str(true_card).lower().strip()
                if pc_low == tc_low:
                    reward += 12.0
                else:
                    reward -= 25.0

        if not comp_text.strip().lower().startswith("card:"):
            reward -= 10.0

        if "\n" in comp_text.strip():
            reward -= 8.0

        if "```" in comp_text or "print(" in comp_text:
            reward -= 20.0

        rewards.append(float(reward))

    return rewards
