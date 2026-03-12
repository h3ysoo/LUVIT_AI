"""
Luvit AI Coach — QLoRA Fine-Tuning Script
=========================================
Run this on Google Colab (free A100 GPU) or any GPU with 16GB+ VRAM.

Setup in Colab:
  !pip install transformers peft trl bitsandbytes accelerate datasets

Then upload your training_data.jsonl and run this script.
"""

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ── 2. CONFIG ─────────────────────────────────────────────────────────────────
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Alternative if you don't have LLaMA access:
# BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./luvit-coach-adapter"
MAX_SEQ_LENGTH = 2048

# ── 3. LOAD MODEL IN 4-BIT (QLoRA) ───────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 4. LORA CONFIGURATION ────────────────────────────────────────────────────
# These target layers are where coaching style/knowledge gets injected
lora_config = LoraConfig(
    r=16,               # Rank — higher = more expressive but more memory
    lora_alpha=32,      # Scaling factor (usually 2x rank)
    target_modules=[    # Attention + MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~1-2% of parameters trainable (~80-100M params)

# ── 5. LOAD DATASET ───────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded {len(dataset)} training examples")

# ── 6. TRAINING ARGUMENTS ────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # Effective batch size = 8
    warmup_steps=50,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",           # Memory-efficient optimizer
    lr_scheduler_type="cosine",
    report_to="none",                   # Set to "wandb" if you want tracking
)

# ── 7. TRAINER ────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field=None,            # We use "messages" format (ChatML)
    packing=True,                       # Pack short examples together = faster
)

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
print("Starting fine-tuning... ☕ This takes ~30-60 min on Colab A100")
trainer.train()

# ── 9. SAVE ADAPTER ──────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA adapter saved to {OUTPUT_DIR}")
print("   Adapter size is only ~50-100MB — upload to HuggingFace Hub!")

# ── 10. QUICK TEST ────────────────────────────────────────────────────────────
from peft import PeftModel
from transformers import pipeline

# Reload for inference
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model_merged = PeftModel.from_pretrained(base, OUTPUT_DIR)

pipe = pipeline("text-generation", model=model_merged, tokenizer=tokenizer)

test_messages = [
    {"role": "system", "content": "You are Lucia, a tough-love personal trainer on the Luvit fitness app."},
    {"role": "user", "content": "Hi! Can you make me a weekly programme?"}
]

result = pipe(test_messages, max_new_tokens=200)
print("\n🤖 Test output:")
print(result[0]["generated_text"][-1]["content"])
