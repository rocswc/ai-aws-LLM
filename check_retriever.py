from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "/opt/dlami/nvme/chroma_store"
COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"}
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}
)

docs = retriever.get_relevant_documents(
    "원당배수지 설치 목적"
)

print(f"검색된 문서 수: {len(docs)}\n")

for i, d in enumerate(docs, 1):
    print(f"[문서 {i}]")
    print(d.page_content[:500])
    print("-" * 80)

