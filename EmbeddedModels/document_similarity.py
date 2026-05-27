from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Initialize Hugging Face Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
documents = [
    "AI is transforming the world",
    "Machine learning is a subset of AI",
    "Python is a programming language"
]

# Query
query = "What is artificial intelligence?"

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Get best match
index, best_score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

# Output
print("Query:")
print(query)

print("\nMost Similar Document:")
print(documents[index])

print("\nSimilarity Score:")
print(round(float(best_score), 4))