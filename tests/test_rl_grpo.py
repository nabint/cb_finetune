import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def format_data(example):
    """Format the JSONL data into a prompt for the model"""
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
    }

def read_jsonl_data(file_path):
    """Read data from JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        continue
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

class CallbreakModel:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", finetuned_dir="./callbreak_agent/trained_model/callbreak_rl_grpo"):
        """Initialize the Callbreak model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.finetuned_dir = finetuned_dir
        
        print(f"Using device: {self.device}")
        print("Loading model...")
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load finetuned weights
        try:
            self.model = PeftModel.from_pretrained(self.base_model, finetuned_dir)
            self.model = self.model.merge_and_unload()
            print("Finetuned model loaded successfully")
        except Exception as e:
            print(f"Error loading finetuned model: {e}")
            print("Using base model instead")
            self.model = self.base_model
        
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")

    def generate_response(self, prompt: str) -> str:
        """Generate response from the model"""
        text = f"Instruction:\n{prompt}\n\nResponse:\n"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Response:" in decoded:
            decoded = decoded.split("Response:")[-1].strip()
        
        return decoded

def process_jsonl_data(jsonl_file_path, model_instance):
    """Process JSONL data and generate responses"""
    # Read JSONL data
    print(f"Reading data from {jsonl_file_path}...")
    data = read_jsonl_data(jsonl_file_path)
    
    if not data:
        print("No data found or error reading file")
        return
    
    print(f"Loaded {len(data)} examples from JSONL file")
    
    # Process each example
    results = []
    
    for i, example in enumerate(data):
        print(f"\nProcessing example {i+1}/{len(data)}:")
        print(f"Game state: {example}")
        
        # Format the data
        formatted_data = format_data(example)
        
        # Generate response
        response = model_instance.generate_response(formatted_data["prompt"])

        if i > 100:
            break
        
        # Store result
        result = {
            "input": example,
            "prompt": formatted_data["prompt"],
            "model_response": response,
            "expected_card": example.get("best_card_to_throw", ""),
            "expected_reason": example.get("reason", "")
        }
        results.append(result)
        
        print("Model Response:")
        print(response)
        print("Expected: {example.get('best_card_to_throw', '')} - {example.get('reason', '')}")
        print("-" * 80)
    
    return results

def save_results(results, output_file="results.json"):
    """Save results to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def interactive_mode(model_instance):
    """Interactive mode for testing individual examples"""
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Enter game state manually or type 'exit' to quit")
    
    while True:
        try:
            print("\nEnter game state (or 'exit' to quit):")
            user_input = input().strip()
            
            if user_input.lower() == 'exit':
                break
            
            if user_input:
                try:
                    # Try to parse as JSON
                    game_state = json.loads(user_input)
                    formatted_data = format_data(game_state)
                    response = model_instance.generate_response(formatted_data["prompt"])
                    
                    print(f"\nModel Response:")
                    print(response)
                except json.JSONDecodeError:
                    print("Invalid JSON format. Please enter a valid JSON game state.")
                except Exception as e:
                    print(f"Error processing input: {e}")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break

def create_sample_jsonl():
    """Create a sample JSONL file for testing"""
    sample_data = [
        {
            "discarded_pile": ["10 of Hearts", "Ace of Hearts", "7 of Hearts", "4 of Hearts"],
            "current_thrown_card": [],
            "leading_card": "",
            "best_card_to_throw": "9 of Spades",
            "legal_cards": ["Jack of Spades", "9 of Spades", "8 of Spades", "5 of Spades", "4 of Spades", "2 of Spades", "5 of Hearts", "3 of Hearts", "2 of Hearts", "Ace of Clubs", "10 of Clubs", "2 of Diamonds"],
            "reason": "Expert leads with 9 of Spades as a cautious contest strong enough to apply pressure but not wasting an Ace/King prematurely."
        },
        {
            "discarded_pile": ["King of Clubs", "Queen of Clubs"],
            "current_thrown_card": ["Jack of Clubs"],
            "leading_card": "Jack of Clubs",
            "best_card_to_throw": "Ace of Clubs",
            "legal_cards": ["Ace of Clubs", "10 of Clubs", "9 of Clubs", "2 of Hearts", "3 of Hearts"],
            "reason": "Play Ace of Clubs to win the trick since opponent led with Jack of Clubs."
        }
    ]
    
    with open("sample_callbreak_data.jsonl", "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print("Sample JSONL file created: sample_callbreak_data.jsonl")

if __name__ == "__main__":
    # Configuration
    JSONL_FILE_PATH = "ai_polished_reasons.jsonl"  # Update this path
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    FINETUNED_DIR = "./callbreak_agent/trained_model/best_card_lora"
    
    # Initialize model
    try:
        callbreak_model = CallbreakModel(BASE_MODEL, FINETUNED_DIR)
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)
    
    # Menu system
    while True:
        print("\n" + "="*50)
        print("CALLBREAK AI AGENT")
        print("="*50)
        print("1. Process JSONL file")
        print("2. Interactive mode")
        print("3. Create sample JSONL file")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Process JSONL file
            results = process_jsonl_data(JSONL_FILE_PATH, callbreak_model)
            if results:
                save_results(results)
        
        elif choice == "2":
            # Interactive mode
            interactive_mode(callbreak_model)
        
        elif choice == "3":
            # Create sample JSONL file
            create_sample_jsonl()
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-4.")