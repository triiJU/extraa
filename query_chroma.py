import os
import requests
import json
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Initialize ChromaDB with HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_papers(query, top_k=3):
    """Search ChromaDB for relevant research papers and generate answers using GroqCloud."""
    try:
        # Fetch relevant documents from ChromaDB
        docs = db.similarity_search(query, k=top_k)

        if not docs:
            return [], "No relevant papers found."

        # Truncate content for readability and token limits
        context = "\n\n".join([doc.page_content[:500] for doc in docs])  # Truncate to 500 characters
        llm_input = f"Context: {context}\n\nQuestion: {query}"

        # Generate output using GroqCloud
        response = groq_client.complete(
            engine="llama-2-7b",  # or "mistral-7b" depending on your choice
            prompt=llm_input,
            max_tokens=500
        )

        generated_text = response.choices[0].text.strip()

        return docs, generated_text
    except Exception as e:
        return [], f"Error during query: {e}"
