from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # for LoRA-only scenarios
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "qwen2.5-0.5b-instruct"

adapter_a_dir = "./outputs/final_model"
adapter_b_dir = "./outputs/optimized_model"


INFERENCE_MODE = "merge"  # change to "compose" or "switch" as needed

# Active adapter to use in "switch" mode
ACTIVE_ADAPTER = "domain_a"  # or "domain_b"

# ===== Load tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    # prefer to set pad_token to eos_token id/string
    tokenizer.pad_token = tokenizer.eos_token

# ===== Load base model =====
# Note: using device_map="auto" will place model on GPU if available
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Wrap with PEFT (load first adapter via from_pretrained)
model = PeftModel.from_pretrained(
    base_model,
    adapter_a_dir,
    adapter_name="domain_a",
)

# Load second adapter (frozen for inference)
model.load_adapter(
    adapter_b_dir,
    adapter_name="domain_b",
    is_trainable=False,
)

# ===== Activate adapters according to mode =====
merged_inference = False

if INFERENCE_MODE == "switch":
    # Use exactly one adapter
    model.set_adapter(ACTIVE_ADAPTER)
    merged_inference = False

elif INFERENCE_MODE == "compose":
    # Compose/activate multiple adapters simultaneously.
    # Many PEFT implementations support setting active adapters via `active_adapters`.
    # Avoid calling set_adapter with a list (that raises the TypeError you saw).
    # Instead set `active_adapters` to a list of adapter names.
    # (This should be supported by the PEFT runtime; if not available, a different
    # approach is required depending on the peft version.)
    try:
        model.active_adapters = ["domain_a", "domain_b"]
    except Exception:
        # fallback: set one adapter active (best-effort). This fallback won't compose,
        # but avoids crashing on older PEFT versions.
        model.set_adapter("domain_a")
    merged_inference = False

elif INFERENCE_MODE == "merge":
    # Merge both adapters into the base weights, then unload PEFT modules.
    # Do NOT call set_adapter() with a list (that caused your TypeError).
    # Instead call merge_and_unload with the adapter names list directly.
    # After merging, move the underlying model to the desired device.
    model = model.merge_and_unload(adapter_names=["domain_a", "domain_b"])
    # The returned object is a regular Transformers model (no PEFT wrappers).
    # Move to target device
    model.to(device)
    merged_inference = True

else:
    raise ValueError("INFERENCE_MODE must be one of: 'switch', 'compose', 'merge'")

model.eval()

# If not fully merged, ensure model is on the target device (PeftModel may already
# have device_map placement; this is a best-effort move for CPU/GPU single-device).
try:
    model.to(device)
except Exception:
    # some PeftModel/transformers configs with device_map="auto" may refuse .to()
    pass

# ===== Helper / generation function =====
def generate_response(prompt: str) -> str:
    # keep your prompt wrapper
    text = f" Instruction:\n{prompt}\n\n Response:\n"
    inputs = tokenizer(text, return_tensors="pt")
    # move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.01,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract the portion after " Response:" if present
    if " Response:" in decoded:
        decoded = decoded.split(" Response:")[-1].strip()
    return decoded

# ===== Example main =====
if __name__ == "__main__":
    print("Model loaded. Type 'exit' to quit.\n")
    print(f"Mode: {INFERENCE_MODE} | Merged: {merged_inference}\n")

    agent_context = """
        Role: Callbreak-only move selector.

        Scope constraint:
        - Only reason about the Callbreak card game. If the request is outside Callbreak or lacks a valid game state, reply exactly: "Cannot answer: non-Callbreak or insufficient game state."
        - Retain previously learned Callbreak rules/strategy; apply new house rules if given; do not invent rules.

        Decision task (output format):
        - Output exactly two lines, nothing else:
        Play: <one legal card from hand>
        Reason: <short, rule-based rationale (<=20 words)>

        Legality and validation:
        - Must follow suit if possible; if void, decide between trumping (spades) or safe discard.
        - Ensure the chosen card is legal given the lead suit and the provided hand.

        Implicit reasoning factors (do not print these):
        - Table position (lead/2nd/3rd/4th), bids vs tricks taken, trump count/management, entries, card counting.
        - Opponents’ voids and revealed high cards; risk control to meet bid; avoid unnecessary overtricks when risky.
        - Endgame: cash sure winners, preserve entries, avoid blocking suits.

        Assumptions:
        - Spades are trump unless the input specifies otherwise.
        - If critical info is missing, state one brief assumption on a third line prefixed with "Assumption:" and keep it to <=10 words.

        Style and safety:
        - Deterministic, concise, Callbreak-only. No meta-talk, no extra lines, no non-Callbreak content.
    """


    q_input = """
        Cards already played in the round (in order): 2 of Clubs, 8 of Clubs Leading card: 8 of Clubs. 
        My legal hand cards are: K of Clubs, A of Clubs, 10 of Clubs. 
        I am player number 3 in this trick. What is the best card to play and why? 
        Please give the card to play, the professional strategy name, and a clear reason for this choice (what the strategy achieves)
    """

    response = generate_response(agent_context + q_input)
    print("\nLLM Response:\n", response, "\n")
