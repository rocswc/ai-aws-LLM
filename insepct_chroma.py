import chromadb
from sentence_transformers import SentenceTransformer

# ===============================
# 설정 (네 ingest 코드와 동일해야 함)
# ===============================
CHROMA_DIR = "/opt/dlami/nvme/chroma_store_hybrid"
COLLECTION_NAME = "pdf_hybrid_collection"
EMBED_MODEL = "intfloat/multilingual-e5-large"

TOP_K = 5

# ===============================
# 임베더 (GPU)
# ===============================
embedder = SentenceTransformer(EMBED_MODEL, device="cuda")

# ===============================
# Chroma 연결
# ===============================
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

print("[INFO] collection size:", collection.count())

# ===============================
# 검색 함수
# ===============================
def search(query: str):
    query_emb = embedder.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    print("\n==============================")
    print("QUERY:", query)
    print("==============================")

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]

        print(f"\n--- Rank {i+1} ---")
        print("Distance:", round(dist, 4))
        print("Page:", meta.get("page"))
        print("Source:", meta.get("source"))
        print("Text:")
        print(doc[:1000])  # 너무 길면 자르기

# ===============================
# 테스트
# ===============================
if __name__ == "__main__":
    while True:
        q = input("\n질문 입력 (exit 입력시 종료): ").strip()
        if q.lower() == "exit":
            break
        search(q)

