import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# =========================
# 기본 설정
# =========================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "/opt/dlami/nvme/sft_train.jsonl"
OUTPUT_DIR = "/opt/dlami/nvme/sft_lora_qwen"

os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"

# =========================
# 토크나이저 / 모델
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)

# =========================
# LoRA 설정
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()

model.enable_input_require_grads()

model.print_trainable_parameters()

# =========================
# 데이터 로드
# =========================
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

# =========================
# 🔥 핵심: text 필드 생성
# =========================
def build_text(example):
    return {
        "text": (
            f"<|system|>\n{example['instruction']}\n"
            f"<|user|>\n{example['input']}\n"
            f"<|assistant|>\n{example['output']}"
        )
    }

dataset = dataset.map(build_text, remove_columns=dataset.column_names)

# =========================
# 학습 파라미터
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # 실효 batch = 16
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
)

# =========================
# SFT Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# =========================
# 학습 시작
# =========================
trainer.train()

# =========================
# 저장
# =========================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ SFT LoRA 학습 완료:", OUTPUT_DIR)

