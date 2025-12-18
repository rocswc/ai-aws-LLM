# save as: inspect_chroma_search.py

import chromadb
from sentence_transformers import SentenceTransformer

# ===============================
# 설정 (네 ingest 코드와 100% 동일)
# ===============================
CHROMA_DIR = "/opt/dlami/nvme/chroma_store_hybrid"
COLLECTION_NAME = "pdf_hybrid_collection"
EMBED_MODEL = "intfloat/multilingual-e5-large"

TOP_K = 5

# ===============================
# 임베더 (GPU)
# ===============================
embedder = SentenceTransformer(
    EMBED_MODEL,
    device="cuda"
)

# ===============================
# Chroma 연결
# ===============================
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

print("[INFO] collection name :", COLLECTION_NAME)
print("[INFO] collection size :", collection.count())

# ===============================
# 검색 함수
# ===============================
def search(query: str):
    print("\n[QUERY]", query)

    # 🔥 핵심: E5 모델 query prefix
    query_emb = embedder.encode(
        [f"query: {query}"],
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"\n--- RESULT {i+1} ---")
        print("[META]", meta)
        print("[DOC]", doc[:500], "...")


# ===============================
# 테스트 실행
# ===============================
if __name__ == "__main__":
    search("2019년 고양시의 녹지지역 면적은 얼마인가요?")

