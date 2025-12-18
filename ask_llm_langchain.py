# save as: ask_llm_langchain.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# =========================
# 1. 환경/경로 설정
# =========================
PERSIST_DIR = "/opt/dlami/nvme/chroma_store"
COLLECTION_NAME = "pdf_collection"

# IMPORTANT:
# 너는 PDF 임베딩을 intfloat/multilingual-e5-large로 만들었으니,
# 질의(검색)도 반드시 같은 임베딩 모델을 써야 함.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# A10G 24GB에 안전하게 올라가는 7B 모델로 고정 (OOM 방지)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

QUERY = """
다음 항목별로 답해줘.

1. 원당배수지 설치 목적
2. 원당배수지 위치
3. 도시·군관리계획 입안권자

각 항목은 제공된 문서에 명확한 근거가 있을 때만 답해줘.
문서에 해당 정보가 없으면 반드시
"제공된 문서에 해당 정보가 없습니다."
라고 써.
""".strip()

# HuggingFace 캐시를 nvme로 강제 (root 디스크 꽉 차는 거 방지)
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"

# =========================
# 2. 임베딩 로드 (GPU)
# =========================
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 32},  # 검색용 임베딩 속도/안정
)

# =========================
# 3. Chroma 벡터스토어 로드
# =========================
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},  # 근거 문서 넉넉히
)

# =========================
# 4. LLM 로드 (OOM 방지 세팅)
# =========================
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda:0",          # 단일 GPU 고정
    low_cpu_mem_usage=True,       # 로딩 메모리 절약
)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,           # 너무 크게 잡으면 느려지고 VRAM/KV 캐시 증가
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=gen_pipeline)

# =========================
# 5. 프롬프트 (항목별 강제)
# =========================
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "너는 한국어 행정 문서 질의 AI다.\n"
        "반드시 [문서] 근거로만 답해라. 추측 금지.\n"
        "각 항목은 아래 형식 그대로 출력해라.\n"
        "근거가 없으면 해당 항목 값으로 정확히 다음 문장만 출력해라: 제공된 문서에 해당 정보가 없습니다.\n"
        "\n"
        "[문서]\n"
        "{context}\n"
        "\n"
        "[질문]\n"
        "{question}\n"
        "\n"
        "[답변 형식]\n"
        "1. 원당배수지 설치 목적: (근거 있으면 내용 / 없으면 제공된 문서에 해당 정보가 없습니다.)\n"
        "2. 원당배수지 위치: (근거 있으면 내용 / 없으면 제공된 문서에 해당 정보가 없습니다.)\n"
        "3. 도시·군관리계획 입안권자: (근거 있으면 내용 / 없으면 제공된 문서에 해당 정보가 없습니다.)\n"
    ),
)

# =========================
# 6. RetrievalQA 체인
# =========================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# =========================
# 7. 실행
# =========================
if __name__ == "__main__":
    result = qa_chain.invoke({"query": QUERY})

    print("\n====================")
    print("Q:")
    print(QUERY)
    print("--------------------")
    print("A:")
    print(result["result"].strip())
    print("====================\n")

    print("Source Documents\n")
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"--- 문서 {i} ---")
        print(doc.page_content[:600])
        print("metadata:", doc.metadata)
        print()

