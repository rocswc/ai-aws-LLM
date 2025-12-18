# save as: embed_pdf_to_chroma.py

import sys
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# =========================
# 1. 경로 설정
# =========================
PDF_PATH = "/home/ubuntu/pdf/input.pdf"
PERSIST_DIR = "/opt/dlami/nvme/chroma_store"
COLLECTION_NAME = "pdf_collection"

CHUNK_SIZE = 300
OVERLAP = 50

# =========================
# 2. PDF 텍스트 추출
# =========================
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# =========================
# 3. 텍스트 청킹
# =========================
def chunk_text(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# =========================
# 4. 임베딩 생성 (safetensors, GPU)
# =========================
def embed_chunks(chunks):
    model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        device="cuda"
    )

    # e5 모델은 prefix 필요
    chunks = [f"passage: {c}" for c in chunks]

    embeddings = model.encode(
        chunks,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings.tolist()

# =========================
# 5. ChromaDB 저장
# =========================
def save_to_chroma(chunks, embeddings):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"doc_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

# =========================
# 6. 실행
# =========================
if __name__ == "__main__":
    print("PDF 텍스트 추출 중...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("텍스트 청킹 중...")
    chunks = chunk_text(raw_text, CHUNK_SIZE, OVERLAP)
    print(f"총 청크 수: {len(chunks)}")

    print("임베딩 생성 중...")
    embeddings = embed_chunks(chunks)

    print("ChromaDB 저장 중...")
    save_to_chroma(chunks, embeddings)

    print("완료")

