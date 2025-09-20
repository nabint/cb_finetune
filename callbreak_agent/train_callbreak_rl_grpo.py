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
raw = load_dataset("json", data_files={"train": "ai_polished_reasons.jsonl"})["train"]


def format_data(example):
    return {
        "prompt": f"""
        ### ROLE: You are a Callbreak expert.
        ### LANGUAGE: You must answer in ENGLISH only.
        ### TASK: Decide the single best card to throw and give one short reason.

        Here is the state:
        Discarded Pile: {example["discarded_pile"]}
        Current Round Cards Played: {example["current_thrown_card"]}
        Leading Card: {example["leading_card"]}
        Legal Cards: {example["legal_cards"]}

        ### STRICT OUTPUT FORMAT (exactly two lines):
        Card: <exact best card to throw from the Legal Cards above>
        Reason: <short English reason>

        Do NOT output code, commentary, or anything else.
        """,
        "best_card_to_throw": example.get("best_card_to_throw", ""),
        "reason": example.get("reason", ""),
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
lora_path = "./callbreak_agent/trained_model/best_card_lora"
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
    Compute rewards based on model completions vs. ground truth.
    All dataset fields are passed as named parameters.
    """

    # Normalize completions
    norm_completions = normalize_texts(completions, tokenizer)

    # Broadcast ground truth arrays to match completions
    true_cards = broadcast(best_card_to_throw, len(norm_completions))
    true_reasons = broadcast(reason, len(norm_completions))

    rewards = []
    for i, (comp_text, true_card, true_reason) in enumerate(
        zip(norm_completions, true_cards, true_reasons)
    ):
        reward = 0.0
        comp_text_l = comp_text.lower()
        pred_card = extract_card(comp_text)

        # 1. Card matching
        if pred_card and true_card:
            tc = str(true_card).lower()
            pc = pred_card.lower()
            reward += 5.0 if pc == tc else (3.0 if tc in pc else 0)

        if random.randint(0, 1) == 0:
            # print("-----------------------")
            # print(prompts[i])
            if pred_card and true_card:
                print(
                    f"TC {str(true_card).lower()} PC {pred_card.lower()} EQ: {pred_card.lower() == str(true_card).lower()}"
                )
                # print(f"Dataset Throw Card: {true_card} | Model Throw Card: {pred_card}\n")
                print(f"Completion text:{comp_text_l}\n")
            else:
                print(f"Empty pred card {comp_text_l}\n")

        # 2. Reason mention bonus
        if true_reason and str(true_reason).lower() in comp_text_l:
            reward += 2.0

        # 3. Completion length
        word_count = len(comp_text_l.split())
        if word_count > 45:   # too long
            reward -= 3.0
        elif 15 <= word_count <= 45:  # acceptable
            reward += 0.5
        elif word_count < 5:  # too short
            reward -= 2.0
        else:  # nice and concise
            reward += 1.0


        # 4. Card-game vocabulary bonus
        reward += sum(
            0.2
            for term in ["trump", "spades", "trick", "lead", "follow"]
            if term in comp_text_l
        )

        if not pred_card:
            reward -= 2.0

        if not comp_text.strip().lower().startswith("card:"):
            reward -= 5.0 
        if "reason:" not in comp_text.lower():
            reward -= 5.0
        if "```" in comp_text or "print(" in comp_text:
            reward -= 10.0
        if any("\u4e00" <= ch <= "\u9fff" for ch in comp_text):
            reward -= 5.0

        rewards.append(float(reward))

    return rewards


# Trainer

grpo_config = GRPOConfig(
    learning_rate=5e-6,
    num_generations=4,
    max_steps=200,
    loss_type="grpo",
    epsilon=0.2,
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
    save_path = "./callbreak_agent/trained_model/callbreak_rl_grpo"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")
except Exception as e:
    print(f"Save failed: {e}")

print("Script completed.")
