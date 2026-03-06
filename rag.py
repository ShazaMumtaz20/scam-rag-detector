from langchain_community.vectorstores import FAISS          # fixed
from langchain_huggingface import HuggingFaceEmbeddings     # fixed
from openai import OpenAI

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)  # fixed

# Initialize OpenAI client
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def retrieve_similar(message, k=2):  # reduced from 3 to 2
    docs = db.similarity_search(message, k=k)
    return [d.page_content[:200] for d in docs]  # trim each chunk to 200 chars

def analyze_scam(message):
    similar = retrieve_similar(message)
    context = "\n".join(similar)

    prompt = f"""
You are a cybersecurity assistant.

User message:
{message}

Similar scam patterns:
{context}

Analyze:
1. Is this message a scam?
2. Risk level (Low / Medium / High)
3. Explain why in simple words.
"""

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",  # smaller model, higher TPM limit
    messages=[{"role": "user", "content": prompt}]
)

    return response.choices[0].message.content