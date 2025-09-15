from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths - Updated to load the fully trained model directly
finetuned_dir = "./outputs/final_model"

# Load the fully trained model directly (no base model + PEFT needed)
model = AutoModelForCausalLM.from_pretrained(
    finetuned_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Match training precision
    trust_remote_code=True
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
        You are an expert Callbreak game strategist.
        I will provide the current game state, including:
        - Cards already played in the round (if any)
        - Your legal hand cards

        Your task:

        Identify the single best card to play next from my hand, given the full context.

        Explain clearly and concisely why this is the best choice based on strategy.
    """

    q_input = """
        Game context:
            - Opponent has already played the 2 of Diamonds and the 5 of Diamonds.x
            - My available options are the ace of Diamonds and the Queen of Diamonds. I can't play any other card

        Question:
            Which card should I play in this situation for the best outcome?
    """
    response = generate_response(agent_context + q_input)
    print("\nLLM Response:\n", response, "\n")

    # while True:
    #     user_input = input("Enter your prompt: ")
    
    #     if user_input.strip().lower() == "exit":
    #         print("Exiting...")
    #         break

    #     response = generate_response(user_input)
    #     print("\nLLM Response:\n", response, "\n")