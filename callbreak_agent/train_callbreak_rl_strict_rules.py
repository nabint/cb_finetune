from trl import GRPOTrainer, GRPOConfig

import traceback
from callbreak_agent.utils import (
    load_and_prepare_dataset,
    load_model_and_tokenizer,
    reward_function,
)


def train_model(base_model, tokenizer, dataset, save_path):
    grpo_config = GRPOConfig(
        learning_rate=5e-5,
        num_generations=4,
        max_steps=100,
        loss_type="grpo",
        epsilon=0.2,
        per_device_train_batch_size=4,
        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=base_model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[reward_function],
        gen_kwargs={
            "max_new_tokens": 64,
            "do_sample": False, # if False, it will be greedy
            "temperature": 0.3, # high temp high randomness
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        },
    )

    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()

    try:
        base_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")
    except Exception as e:
        print(f"Save failed: {e}")


if __name__ == "__main__":
    dataset = load_and_prepare_dataset("ai_polished_reasons_4.jsonl")
    print(f"Dataset size: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(
        lora_path="./callbreak_agent/trained_model/callbreak_rl_grpo_3"
    )

    train_model(
        model,
        tokenizer,
        dataset,
        save_path="./callbreak_agent/trained_model/callbreak_rl_grpo_4",
    )

    print("Script completed.")
