import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LORA_DIR = "/opt/dlami/nvme/sft_lora_qwen"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

prompt = """다음 행정 문서를 근거로 질문에 답하시오.

질문: 2019년 고양시의 녹지지역 면적은 얼마인가요?

문서:
2019년 고양시내 녹지지역이 전체면적의 56.77%로 가장 넓은 면적을 차지하고 있으며
녹지지역 면적은 149.48km²로 조사되었다.

답변:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

generated = outputs[0][inputs["input_ids"].shape[-1]:]
print("📌 모델 답변:")
print(tokenizer.decode(generated, skip_special_tokens=True))

