from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths - Updated to load the fully trained model directly
finetuned_dir = "./outputs/optimized_model"

# Load the fully trained model directly (no base model + PEFT needed)
model = AutoModelForCausalLM.from_pretrained(
    finetuned_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Match training precision
    trust_remote_code=True,
)
model.eval()

# Load tokenizer from the saved model directory
tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Function to generate responses using the chat template
def generate_response(prompt: str) -> str:
    # Use the same system prompt as during training
    system_prompt = "You are an expert callbreak card game master with deep knowledge of rules, strategies, scoring, and advanced gameplay techniques. Provide comprehensive, accurate, and insightful responses about callbreak gameplay, rules clarifications, strategic advice, and game mechanics. Always be precise and educational in your explanations."

    # Format using the chat template used during training
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    if "<|im_start|>assistant\n" in decoded:
        decoded = decoded.split("<|im_start|>assistant\n")[-1].strip()

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
        My legal hand cards are: K of Clubs, A of Clubs, 10 of Clubs. 
        I am player number 3 in this trick. What is the best card to play and why? 
        Please give the card to play, the professional strategy name, and a clear reason for this choice (what the strategy achieves)
    """
    response = generate_response(agent_context + q_input)
    print("\nLLM Response:\n", response, "\n")