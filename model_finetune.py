import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, get_cosine_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# Force CUDA and optimize settings
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# Load and prepare dataset
try:
    dataset = load_dataset("json", data_files={"train": "data.jsonl"})["train"]
    dataset = dataset.train_test_split(test_size=0.05, seed=42)  # Smaller test split for more training data
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Load model with full precision for maximum performance
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="qwen2.5-0.5b-instruct",
        max_seq_length=4096,
        dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
        load_in_4bit=False,    # Disable quantization for full training
        device_map="auto"      # Automatic device mapping
    )
    
    # Convert model to full training mode (remove any quantization)
    model = model.to(torch.bfloat16)
    print("Model loaded in full precision mode")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Enable full parameter training instead of LoRA
try:
    # Unfreeze all parameters for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print("Model configured for full parameter training")
except Exception as e:
    print(f"Error configuring model: {e}")
    exit(1)

# Advanced formatting with enhanced system prompt
def formatting_prompts_func(examples):
    texts = []
    system_prompt = """You are an expert callbreak card game master with deep knowledge of rules, strategies, scoring, and advanced gameplay techniques. Provide comprehensive, accurate, and insightful responses about callbreak gameplay, rules clarifications, strategic advice, and game mechanics. Always be precise and educational in your explanations."""
    
    for prompt, response in zip(examples["prompt"], examples["response"]):
        # Enhanced chat template formatting
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

# Apply formatting
train_dataset = dataset["train"].map(formatting_prompts_func, batched=True, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["test"].map(formatting_prompts_func, batched=True, remove_columns=dataset["test"].column_names)

print(f"Formatted datasets: {len(train_dataset)} train, {len(eval_dataset)} eval")

# High-performance training configuration
training_args = TrainingArguments(
    output_dir="./outputs",
    
    # Batch size and gradient settings - optimized for performance
    per_device_train_batch_size=8,        # Larger batch size
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,         # Effective batch size = 16
    
    # Training duration - more training for better performance
    num_train_epochs=15,                   # More epochs for deeper learning
    max_steps=-1,
    
    # Learning rate optimization
    learning_rate=3e-5,                    # Optimal LR for full fine-tuning
    warmup_ratio=0.1,                      # 10% warmup
    lr_scheduler_type="cosine_with_restarts",  # Advanced scheduler
    
    # Precision and optimization
    bf16=True,                             # Better numerical stability
    fp16=False,
    tf32=True,                             # Enable TF32 on Ampere GPUs
    
    # Regularization for better generalization
    weight_decay=0.01,
    max_grad_norm=1.0,                     # Gradient clipping
    
    # Evaluation and saving
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Logging and monitoring
    logging_steps=25,
    report_to="none",
    
    # Advanced optimizations
    dataloader_num_workers=4,              # Parallel data loading
    group_by_length=True,                  # Group similar lengths for efficiency
    length_column_name="length",
    remove_unused_columns=False,
    
    # Memory and performance optimizations
    dataloader_pin_memory=True,
    ignore_data_skip=True,
    
    # Disable unnecessary features for performance
    push_to_hub=False,
    hub_model_id=None,
)

# Custom data collator for optimal padding
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
    pad_to_multiple_of=8,  # Optimize for tensor cores
)

# Advanced trainer configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    
    # Performance optimizations
    packing=True,                          # Pack multiple examples
    dataset_num_proc=4,                    # Parallel processing
    
    # Training arguments
    args=training_args,
    data_collator=data_collator,
    
    # Callbacks for advanced training
    callbacks=None,
)

# Custom optimizer for better performance
def create_optimizer():
    """Create optimized AdamW optimizer"""
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.95),                 # Optimized betas
        weight_decay=training_args.weight_decay,
        eps=1e-6,                          # Better numerical stability
    )
    return optimizer

# Override trainer's optimizer
trainer.create_optimizer = create_optimizer

# Additional training optimizations
def setup_training_optimizations():
    """Setup additional optimizations for training"""
    
    # Enable compilation for PyTorch 2.0+ (if available)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='max-autotune')
            print("✅ Model compiled with torch.compile for maximum performance")
    except:
        print("⚠️  torch.compile not available, continuing without compilation")
    
    # Set optimal number of threads
    torch.set_num_threads(8)
    
    # Enable optimized attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("✅ FlashAttention enabled")
    except:
        print("⚠️  FlashAttention not available")

setup_training_optimizations()

# Pre-training validation
print("\n🔍 Pre-training validation...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f}GB")

# Start intensive training
try:
    print("\n🚀 Starting high-performance full model training...")
    print("⚡ Training with full parameter updates for maximum performance")
    
    # Train the model
    trainer.train()
    
    print("\n✅ Training completed successfully!")
    
    # Post-training evaluation
    eval_results = trainer.evaluate()
    print(f"\n📊 Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
        
except Exception as e:
    print(f"❌ Error during training: {e}")
    exit(1)

# Save the fully trained model
try:
    print("\n💾 Saving fully trained model...")
    
    # Save with full precision
    model.save_pretrained(
        "./outputs/final_model",
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained("./outputs/final_model")
    
    # Save training state
    trainer.save_state()
    
    print("✅ Model saved successfully to ./outputs/final_model")
    print("📁 Training state saved for potential resumption")
    
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Training summary
print("\n" + "="*60)
print("🎯 HIGH-PERFORMANCE TRAINING COMPLETE!")
print("="*60)
print("✅ Full parameter fine-tuning completed")
print("✅ Model weights directly updated (no LoRA)")
print("✅ Optimized for maximum performance")
print("✅ Advanced training techniques applied")
print(f"✅ Model ready for deployment at ./outputs/final_model")
print("="*60)