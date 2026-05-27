import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Gemini embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Streamlit page config
st.set_page_config(
    page_title="Similarity Search App",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Document Similarity Search")
st.write("Find the most similar document using Gemini embeddings.")

# Text area for documents
docs_input = st.text_area(
    "Enter documents (one per line):",
    height=200,
    placeholder="""AI is transforming the world
Machine learning is a subset of AI
Python is a programming language"""
)

# Query input
query = st.text_input(
    "Enter your query:",
    placeholder="What is AI?"
)

# Search button
if st.button("Find Most Similar Document"):

    # Validation
    if not docs_input.strip():
        st.warning("Please enter some documents.")

    elif not query.strip():
        st.warning("Please enter a query.")

    else:
        # Convert documents into list
        documents = [
            doc.strip()
            for doc in docs_input.split("\n")
            if doc.strip()
        ]

        try:
            # Generate embeddings
            with st.spinner("Generating embeddings..."):

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

            # Display results
            st.success("Most Similar Document Found!")

            st.subheader("🔎 Query")
            st.write(query)

            st.subheader("📄 Most Similar Document")
            st.write(documents[index])

            st.subheader("📊 Similarity Score")
            st.write(round(float(best_score), 4))

            # Show all scores
            st.subheader("📋 All Similarity Scores")

            for i, score in enumerate(scores):
                st.write(
                    f"Document {i+1}: {round(float(score), 4)}"
                )

        except Exception as e:
            st.error(f"Error: {e}")