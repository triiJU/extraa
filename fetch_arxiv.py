#Here this serves as the backend logic for my project

import arxiv
import chromadb
import requests
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings

# Initializing - HuggingFace 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)


chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="arxiv_papers")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_all_categories():
    """Scrape arXiv to get all available categories dynamically."""
    base_url = "https://arxiv.org"
    response = requests.get(f"{base_url}/archive")
    soup = BeautifulSoup(response.text, "html.parser")
    
    categories = []
    for link in soup.select("a[href^='/list/']"):
        category = link["href"].split("/")[-1]
        categories.append(category)
    return categories

def fetch_category_papers(category, max_results_per_category):
    """Fetch papers for a single arXiv category."""
    search = arxiv.Search(
        query=f"cat:{category}", max_results=max_results_per_category, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return papers

def fetch_all_arxiv_papers(max_results_per_category=5):
    """Fetch papers from all ArXiv categories dynamically using multi-threading."""
    categories = get_all_categories()
    all_papers = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda cat: fetch_category_papers(cat, max_results_per_category), categories)
    for paper_list in results:
        all_papers.extend(paper_list)
    return all_papers

def store_papers(papers):
    """Store paper embeddings in ChromaDB."""
    for paper in papers:
        embedding = model.encode(paper['summary']).tolist()
        collection.add(
            ids=[paper['title']], embeddings=[embedding], metadatas=[{"title": paper['title'], "url": paper['pdf_url']}]
        )
