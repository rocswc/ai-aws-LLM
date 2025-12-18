# save as: generate_qa_from_chroma.py
# 목적: 이미 만들어둔 ChromaDB에서 context를 뽑아 (RAG 근거)
#       로컬 LLM(Qwen)으로 QA를 자동 생성해 jsonl 데이터셋으로 저장

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_json_block(text: str) -> Optional[str]:
    """
    LLM 출력에서 JSON 배열( [ ... ] )만 최대한 뽑아냄
    """
    m = re.search(r"\[\s*\{.*\}\s*\]\s*$", text, re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        return m.group(0)
    return None


def build_llm(model_id: str, device: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )
    return gen


QA_PROMPT = """
너는 "문서 기반 QA 데이터셋" 생성기다.
아래 [문서 컨텍스트]만 근거로 사용해 질문-답변 쌍을 생성해라.
추측/상상 금지.
문서에 근거가 없으면 답변을 정확히 "문서에 해당 정보가 없습니다." 로 작성해라.

요구사항:
- 질문은 사용자가 실제로 할 법한 한국어 질문으로 만들 것
- 답변은 문서 표현을 최대한 보존하며, 짧고 정확하게 작성할 것
- 반드시 JSON 배열만 출력할 것 (설명, 문장, 코드블록 출력 금지)
- 출력 형식은 아래 예시와 동일해야 함
- 총 {n_pairs}개 생성

출력 예시:
[
  {{
    "question": "질문 내용",
    "answer": "답변 내용"
  }}
]

[문서 컨텍스트]
{context}
""".strip()


def generate_pairs(gen, context: str, n_pairs: int) -> List[Dict[str, str]]:
    prompt = QA_PROMPT.format(context=context[:12000], n_pairs=n_pairs)
    out = gen(prompt)[0]["generated_text"]
    out = out.strip()

    jtxt = extract_json_block(out)
    if not jtxt:
        return []

    try:
        data = json.loads(jtxt)
        pairs = []
        for item in data:
            q = clean_text(str(item.get("question", "")))
            a = clean_text(str(item.get("answer", "")))
            if q and a:
                pairs.append({"question": q, "answer": a})
        return pairs
    except Exception:
        return []


def load_existing_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma_dir", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--out", default="qa_dataset.jsonl")

    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--pairs_per_context", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=4000)
    parser.add_argument("--page_context_chunks", type=int, default=4)

    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/opt/dlami/nvme/.cache")

    print("[INIT] Chroma")
    client = chromadb.PersistentClient(path=args.chroma_dir)
    col = client.get_or_create_collection(args.collection)
    total = col.count()
    print(f"[CHROMA] collection={args.collection}, count={total}")

    print("[INIT] LLM")
    gen = build_llm(args.llm_model, args.device, args.max_new_tokens)

    already = load_existing_count(args.out) if args.resume else 0
    if already > 0:
        print(f"[RESUME] existing lines={already} (append mode)")

    fout = open(args.out, "a", encoding="utf-8")

    # 페이지별로 컨텍스트를 "여러 chunk 묶어서" 만들기 위해 버퍼링
    page_buffers: Dict[Any, List[Dict[str, Any]]] = {}

    def flush_page(page_key):
        items = page_buffers.get(page_key, [])
        if not items:
            return 0

        # chunk 몇 개를 합쳐 컨텍스트 구성
        chunks = [it["text"] for it in items][: args.page_context_chunks]
        context = clean_text("\n\n".join(chunks))
        context = context[: args.max_context_chars]

        meta0 = items[0]["meta"] or {}
        page = meta0.get("page", None)
        source = meta0.get("source", None)
        ocr_conf = meta0.get("ocr_conf", None)

        pairs = generate_pairs(gen, context, args.pairs_per_context)
        wrote = 0
        for p in pairs:
            row = {
                "question": p["question"],
                "answer": p["answer"],
                "context": context,
                "page": page,
                "source": source,
                "ocr_conf": ocr_conf,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

        page_buffers[page_key] = []
        return wrote

    written = 0
    offset = 0
    processed = 0

    while offset < total:
        got = col.get(
            include=["documents", "metadatas"],
            limit=args.batch,
            offset=offset,
        )

        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []

        if not docs:
            break

        for text, meta in zip(docs, metas):
            processed += 1
            t = clean_text(text or "")
            if not t:
                continue

            # page 기준으로 묶기. page가 없으면 None으로 묶어서 그냥 흘려보냄
            page_key = (meta or {}).get("page", None)

            if page_key not in page_buffers:
                page_buffers[page_key] = []

            page_buffers[page_key].append({"text": t, "meta": meta})

            # 버퍼가 충분히 차면 flush
            if len(page_buffers[page_key]) >= args.page_context_chunks:
                written += flush_page(page_key)
                if written % 30 == 0 and written > 0:
                    fout.flush()
                    print(f"[PROGRESS] processed_chunks={processed} written_pairs={written}")

        offset += args.batch

    # 남은 페이지 버퍼 flush
    for pk in list(page_buffers.keys()):
        written += flush_page(pk)

    fout.flush()
    fout.close()
    print(f"[DONE] processed_chunks={processed}, written_pairs={written}")
    print(f"[OUT] {args.out}")


if __name__ == "__main__":
    main()

