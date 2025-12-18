# ============================================================
# Hybrid PDF Ingestion Pipeline (B안: VLM Safe + Page Skip)
# OCR: EasyOCR (CPU)
# VLM/SLLM/Embedding: GPU
# Start from page 170
# ============================================================

import os
import re
import json
import uuid
from typing import List, Dict, Any, Optional

import fitz
import numpy as np
from PIL import Image

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForVision2Seq,
    pipeline,
)

import easyocr
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# 0. 환경 변수 (NVMe 캐시)
# ============================================================
NVME = "/opt/dlami/nvme"
os.environ["HF_HOME"] = f"{NVME}/hf_cache"
os.environ["XDG_CACHE_HOME"] = f"{NVME}/.cache"


# ============================================================
# 1. 설정
# ============================================================
PDF_PATH = "/home/ubuntu/pdf/input.pdf"

CHROMA_DIR = f"{NVME}/chroma_store_hybrid"
COLLECTION_NAME = "pdf_hybrid_collection"

EMBED_MODEL = "intfloat/multilingual-e5-large"
SLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLM_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"

RENDER_DPI = 200
MIN_OCR_CHARS = 120
MIN_OCR_CONF = 0.40

CHUNK_SIZE = 900
OVERLAP = 150

# ✅ 170페이지부터 시작
START_PAGE_1IDX = 170
START_PAGE_0IDX = START_PAGE_1IDX - 1  # 169


# ============================================================
# 2. 유틸
# ============================================================
def clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += max(1, CHUNK_SIZE - OVERLAP)
    return chunks


def render_page(page: fitz.Page) -> Image.Image:
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ============================================================
# 3. EasyOCR (CPU)
# ============================================================
def build_ocr():
    return easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)


def run_ocr(reader, img: Image.Image):
    arr = np.array(img)
    results = reader.readtext(arr, detail=1)

    texts, confs = [], []
    for _, text, conf in results:
        texts.append(text)
        confs.append(conf)

    merged = clean_text("\n".join(texts))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return merged, avg_conf


def need_vlm(text: str, conf: float) -> bool:
    return len(text) < MIN_OCR_CHARS or conf < MIN_OCR_CONF


# ============================================================
# 4. SLLM (GPU)
# ============================================================
def build_sllm():
    tokenizer = AutoTokenizer.from_pretrained(SLLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        SLLM_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=512,
        return_full_text=False,
    )


SLLM_PROMPT = """
너는 행정 문서 OCR 텍스트 정제기다.

규칙:
- 원문에 없는 내용 추가 금지
- 홍보/장식 문구 제거
- 표/항목 구조화
- 숫자/연도/단위 보존

JSON만 출력:
{{
  "doc_type": "정책|예산|성과|계획|홍보|기타",
  "title": "",
  "normalized_text": ""
}}

[입력]
{input_text}
"""


def extract_first_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def refine_text(gen, text: str) -> Dict[str, Any]:
    prompt = SLLM_PROMPT.format(input_text=text[:8000])
    out = gen(prompt)[0]["generated_text"]

    json_str = extract_first_json(out)
    if json_str:
        out = json_str

    try:
        data = json.loads(out)
        data["normalized_text"] = clean_text(
            data.get("normalized_text", text)
        )
        data["doc_type"] = data.get("doc_type", "기타") or "기타"
        data["title"] = data.get("title", "") or ""
        return data
    except Exception:
        return {
            "doc_type": "기타",
            "title": "",
            "normalized_text": clean_text(text),
        }


# ============================================================
# 5. VLM (GPU) — SAFE VERSION (B안 핵심)
# ============================================================
def build_vlm():
    processor = AutoProcessor.from_pretrained(VLM_MODEL)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForVision2Seq.from_pretrained(
        VLM_MODEL,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def run_vlm_safe(processor, model, img: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "이 이미지는 한국어 행정 문서 페이지다. 보이는 텍스트만 추출해라."},
        ],
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    try:
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
            )

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return clean_text(text)

    except RuntimeError as e:
        if "cuda" in str(e).lower() or "device-side assert" in str(e).lower():
            print("[VLM-SKIP] CUDA assert → skip this page")
            torch.cuda.empty_cache()
            return ""
        raise


# ============================================================
# 6. Embedding (GPU)
# ============================================================
def build_embedder():
    return SentenceTransformer(EMBED_MODEL, device="cuda")


def embed(embedder, texts: List[str]):
    texts = [f"passage: {t}" for t in texts]
    return embedder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()


# ============================================================
# 7. 메인
# ============================================================
def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    ocr = build_ocr()
    sllm = build_sllm()
    vlm_processor, vlm_model = build_vlm()
    embedder = build_embedder()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION_NAME)

    doc = fitz.open(PDF_PATH)
    total_pages = doc.page_count

    print(f"[PDF] pages={total_pages}")
    print(f"[RUN] start_page={START_PAGE_1IDX} (1-index)")

    for i in range(START_PAGE_0IDX, total_pages):
        try:
            page = doc.load_page(i)
            img = render_page(page)

            ocr_text, conf = run_ocr(ocr, img)
            print(f"[DBG] page={i+1} ocr_len={len(ocr_text)} conf={conf:.3f}")

            # ---------- B안 핵심 ----------
            if need_vlm(ocr_text, conf):
                vlm_text = run_vlm_safe(vlm_processor, vlm_model, img)

                if not vlm_text.strip():
                    print(f"[SKIP] page={i+1} (VLM failed)")
                    continue  # ✅ 이 페이지 완전히 버림
                text = clean_text(ocr_text + "\n" + vlm_text)
                used_vlm = True
            else:
                text = ocr_text
                used_vlm = False
            # --------------------------------

            refined = refine_text(sllm, text)
            chunks = chunk_text(refined["normalized_text"])
            if not chunks:
                print(f"[SKIP] page={i+1} empty_text")
                continue

            vectors = embed(embedder, chunks)

            metadatas = [{
                "page": i + 1,
                "chunk_idx": idx,
                "doc_type": refined["doc_type"],
                "title": refined["title"],
                "used_vlm": used_vlm,
                "ocr_conf": float(conf),
            } for idx in range(len(chunks))]

            col.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=chunks,
                embeddings=vectors,
                metadatas=metadatas,
            )

            print(f"[OK] page={i+1}, chunks={len(chunks)}, used_vlm={used_vlm}")

        except Exception as e:
            print(f"[ERR] page={i+1} :: {type(e).__name__}: {e}")

    print("DONE")


if __name__ == "__main__":
    main()

