from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
# base_model_name = "qwen2.5-0.5b-instruct"
# finetuned_dir = "./outputs/final_model"
base_model_name = "Qwen2.5-1.5B-Instruct"
finetuned_dir = "./callbreak_agent/trained_model/best_card_lora"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load finetuned weights    
model = PeftModel.from_pretrained(base_model, finetuned_dir)
model = model.merge_and_unload()
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate responses
def generate_response(prompt: str) -> str:
    text = f" Instruction:\n{prompt}\n\n Response:\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.01,
            # top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if " Response:" in decoded:
        decoded = decoded.split(" Response:")[-1].strip()

    return decoded

if __name__ == "__main__":
    print("Model loaded. Type 'exit' to quit.\n")
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
        My legal hand cards are: 2 of Spades, K of Hearts, 10 of Hearts. 
        I am player number 3 in this trick. What is the best card to play and why? 
        Please give the card to play, the professional strategy name, and a clear reason for this choice (what the strategy achieves)
    """
    response = generate_response(agent_context + q_input)
    print("\nLLM Response:\n", response, "\n")
