import os
import faiss
from sentence_transformers import SentenceTransformer

# --------- Load documents ----------
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r") as f:
            documents.append(f.read())
    return documents

# --------- Chunking ----------
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# --------- Prepare data ----------
docs = load_documents("data")

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

# --------- Embeddings ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# --------- Vector DB ----------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --------- Retrieval ----------
def retrieve(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# --------- Test query ----------
question = "How many paid leaves do employees get?"
results = retrieve(question)

print("QUESTION:", question)
print("\nRETRIEVED CHUNKS:")
for r in results:
    print("-", r)
