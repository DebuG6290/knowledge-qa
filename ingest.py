import chromadb
import os
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge")

def add_text(text, source_name):
    words = text.split()
    overlap = 50
    chunk_size = 200
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # step forward by 150, not 200
    
    for j, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{source_name}_{j}"]
        )
    print(f"Added {len(chunks)} chunks from '{source_name}'")

# Read all files from articles folder
for filename in os.listdir("articles"):
    if filename.endswith(".txt"):
        with open(f"articles/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
        add_name = filename.replace(".txt", "")
        add_text(text, add_name)