from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load dataset
lines = []
with open("spam", "r", encoding="utf-8") as f:
    for line in f:
        # Split on tab
        parts = line.strip().split("\t")
        if len(parts) == 2:
            label, text = parts
            # Only keep spam messages
            if label.lower() == "spam":
                lines.append(text)

print(f"Loaded {len(lines)} spam messages")

# Split into chunks (RAG-friendly)
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_text("\n".join(lines))

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS vector database
db = FAISS.from_texts(docs, embeddings)

# Save vector DB
db.save_local("vector_db")

print("Vector database created successfully!")
