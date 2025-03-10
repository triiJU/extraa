import os
import requests
import json
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import pysqlite3


sys.modules["sqlite3"] = pysqlite3


# Initialize ChromaDB with HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

@st.cache_resource()
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {model_name}...")
    
    model_path = os.path.join("models", model_name.replace("/", "_"))
    os.makedirs(model_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500)
    return pipe

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to query the model and generate answers
def query_papers(query, top_k=3):
    """Search ChromaDB for relevant research papers and generate answers using the Hugging Face model."""
    try:
        # Fetch relevant documents from ChromaDB
        docs = db.similarity_search(query, k=top_k)

        if not docs:
            return [], "No relevant papers found."

        # Truncate content for readability and token limits
        context = "\n\n".join([doc.page_content[:500] for doc in docs])  # Truncate to 500 characters
        llm_input = f"Context: {context}\n\nQuestion: {query}"

        # Tokenize input and generate output using the Hugging Face model
        inputs = tokenizer(llm_input, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=500)

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return docs, generated_text
    except Exception as e:
        return [], f"Error during query: {e}"
