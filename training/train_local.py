
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- Configuration ---
# We use the 4-bit quantized version directly to save memory during loading
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit" 
DATASET_FILE = "training_dataset_headless.jsonl"
OUTPUT_DIR = "gemma_3_news_analyzer"
MAX_SEQ_LENGTH = 2048

# --- 1. Load Model & Tokenizer (4GB VRAM Optimized) ---
print(f"⏳ Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,             # Auto-detects float16/bfloat16
    load_in_4bit = True,      # CRITICAL for 4GB VRAM
)

# --- 2. Add LoRA Adapters ---
# This makes the model trainable with minimal memory
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,                    # Rank (Keep low for efficiency)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Saves massive VRAM
    random_state = 3407,
)

# --- 3. Prepare Data ---
# We simply concatenate Input + Output because the model needs to learn 
# to generate the Output immediately after seeing the Input.
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

def format_prompts(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        # The training sample is: <ARTICLE>...JSON<eos>
        text = input_text + "\n" + output_text + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)
print(f"✅ Loaded {len(dataset)} training samples.")

# --- 4. Configure Trainer ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # False is safer for VRAM than True
    args = TrainingArguments(
        per_device_train_batch_size = 1,   # Must be 1 for 4GB VRAM
        gradient_accumulation_steps = 8,    # Simulates batch_size = 8
        warmup_steps = 5,
        num_train_epochs = 3.5,              # 2 passes over data (approx 130 steps)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",              # 8-bit optimizer saves memory
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "checkpoints",
    ),
)

# --- 5. Train ---
print("🚀 Starting Training...")
trainer_stats = trainer.train()
print("🎉 Training Complete!")

# --- 6. Export to GGUF (Phase 3) ---
print("📦 Exporting to GGUF (q4_k_m)...")
# This creates the file 'gemma_3_news_analyzer-unsloth.Q4_K_M.gguf'
model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")
print(f"✅ GGUF Model saved to: {OUTPUT_DIR}/{MODEL_NAME.split('/')[-1]}.Q4_K_M.gguf")