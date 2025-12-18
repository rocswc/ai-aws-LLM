import chromadb

# 기존 DB에 연결
client = chromadb.Client()
collection = client.get_collection(name="pdf_collection")

# 저장된 문서 일부 조회
results = collection.get(include=["documents"], limit=5)

print("[저장된 문서 예시]")
for i, doc in enumerate(results["documents"]):
    print(f"{i+1}. {doc[:150]}...\n")

