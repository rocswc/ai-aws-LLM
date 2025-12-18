import json

SRC = "../qa_clean2.jsonl"
DST = "../sft_train.jsonl"

SYSTEM_INSTRUCTION = (
    "다음 행정 문서 내용을 바탕으로 질문에 대해 "
    "사실에 근거하여 간결하고 정확하게 답하시오."
)

with open(SRC, "r", encoding="utf-8") as fin, \
     open(DST, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)

        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        ctx = item.get("context", "").strip()

        if not q or not a or not ctx:
            continue

        sample = {
            "instruction": SYSTEM_INSTRUCTION,
            "input": f"질문: {q}\n\n문서:\n{ctx}",
            "output": a
        }

        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("✅ SFT 포맷 변환 완료:", DST)

