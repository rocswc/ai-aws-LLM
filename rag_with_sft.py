# save as: rag_with_sft_cli.py

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ===============================
# 환경 설정 (네 환경 그대로)
# ===============================
CHROMA_DIR = "/opt/dlami/nvme/chroma_store_hybrid"
COLLECTION_NAME = "pdf_hybrid_collection"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "/opt/dlami/nvme/sft_lora_qwen"

EMBED_MODEL = "intfloat/multilingual-e5-large"
DEVICE = "cuda"

os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"


# ===============================
# Embedding (Chroma와 동일해야 함)
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)


# ===============================
# VectorStore (기존 Chroma 재사용)
# ===============================
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


# ===============================
# SFT + LoRA 모델 로드
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)

# LoRA 어댑터 적용
model.load_adapter(LORA_PATH)
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)


# ===============================
# 🔥 핵심: 수정된 프롬프트
# ===============================
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
너는 행정 문서를 분석하는 전문가다.

규칙:
- 답변은 반드시 제공된 문서(context)에 근거해야 한다.
- 문서에 표나 수치가 있는 경우,
  비교, 계산, 해석을 통해 결론을 도출하는 것을 허용한다.
- 문서에 전혀 근거가 없는 경우에만
  "문서에 해당 정보가 없습니다."라고 답하라.
- 같은 문구를 반복하지 말고 하나의 결론으로 답하라.
- 불필요한 원문 복붙은 금지한다.

[문서]
{context}

[질문]
{question}

[답변]
""".strip(),
)


# ===============================
# RetrievalQA Chain
# ===============================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)


# ===============================
# CLI 인터페이스
# ===============================
def main():
    print("\n✅ RAG + SFT QA 시스템 준비 완료")
    print("질문을 입력하세요. (종료: exit / quit)\n")

    while True:
        query = input("❓ 질문 > ").strip()
        if query.lower() in ("exit", "quit"):
            print("종료합니다.")
            break

        result = qa.invoke({"query": query})

        print("\n📌 답변:")
        print(result["result"])

        print("\n📌 근거 문서:")
        seen_pages = set()
        for doc in result["source_documents"]:
            page = doc.metadata.get("page")
            if page not in seen_pages:
                print(f"- page={page}")
                seen_pages.add(page)

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

