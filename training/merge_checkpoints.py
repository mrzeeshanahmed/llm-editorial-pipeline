import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- 1. CONFIGURATION: EDIT THESE PATHS ---
# Add the paths to your 3 actual checkpoint folders here
CHECKPOINTS = [
    "checkpoints/checkpoint-128",  # Example: Replace with your actual folder names
    "checkpoints/checkpoint-256",
    "checkpoints/checkpoint-448"
]
BASE_MODEL = "unsloth/gemma-3-1b-it-bnb-4bit"
OUTPUT_DIR = "gemma_3_soup_merged"

def merge_lora_soups():
    print(f"🍲 Cooking Model Soup from: {CHECKPOINTS}")

    # Load Base Model
    print("⏳ Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS[-1])

    # Load First Checkpoint (Master)
    print(f"🔹 Loading Master: {CHECKPOINTS[0]}")
    model = PeftModel.from_pretrained(base_model, CHECKPOINTS[0])
    soup_state_dict = model.state_dict()

    # Add other checkpoints
    for ckpt in CHECKPOINTS[1:]:
        print(f"🔹 Blending: {ckpt}")
        temp_model = PeftModel.from_pretrained(base_model, ckpt)
        temp_state_dict = temp_model.state_dict()
        
        # Average the LoRA weights
        for key in soup_state_dict:
            if "lora" in key: 
                soup_state_dict[key] += temp_state_dict[key]
    
    # Divide by number of checkpoints
    n = len(CHECKPOINTS)
    print(f"➗ Averaging weights by {n}...")
    for key in soup_state_dict:
        if "lora" in key:
            soup_state_dict[key] = soup_state_dict[key] / n

    # Save
    print(f"💾 Saving Souped Model to {OUTPUT_DIR}...")
    model.load_state_dict(soup_state_dict)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done! Now convert this folder to GGUF.")

if __name__ == "__main__":
    merge_lora_soups()