from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
base_model_name = "unsloth/qwen2.5-0.5b-instruct"
finetuned_dir = "./outputs/final_model"

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
            temperature=0.7,
            top_p=0.9,
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