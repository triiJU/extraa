# import os
# import arxiv
# import chromadb
# import torch
# import scrapy
# from scrapy.crawler import CrawlerProcess
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFacePipeline
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.docstore.document import Document
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from huggingface_hub import snapshot_download

# # Ensure output directories exist
# base_path = "C:/Users/91748/Documents/arxiv-rag"
# data_path = os.path.join(base_path, "data")
# model_path = os.path.join(base_path, "models")
# os.makedirs(data_path, exist_ok=True)
# os.makedirs(model_path, exist_ok=True)

# # Step 1: Detect CPU or GPU
# def get_device():
#     if torch.cuda.is_available():
#         return "cuda"
#     elif torch.backends.mps.is_available():
#         return "mps"
#     else:
#         return "cpu"

# device = get_device()
# print(f"Using device: {device}")

# # Step 2: Load LLM (Mistral or LLaMA-2)
# def load_llm():
#     model_choices = {"1": "mistralai/Mistral-7B-v0.1", "2": "meta-llama/Llama-2-7b-hf"}
#     choice = input("Choose an LLM: (1) Mistral-7B or (2) LLaMA-2: ")
#     model_name = model_choices.get(choice, "mistralai/Mistral-7B-v0.1")
#     print(f"Loading {model_name}...")

#     model_path_local = os.path.join(model_path, model_name.replace("/", "_"))
#     if not os.path.exists(model_path_local):
#         print("Downloading model...")
#         snapshot_download(repo_id=model_name, local_dir=model_path_local, resume_download=True)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path_local, local_files_only=True)
#     model = AutoModelForCausalLM.from_pretrained(model_path_local, torch_dtype="auto", device_map=device, local_files_only=True)
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500)
#     return HuggingFacePipeline(pipeline=pipe)

# llm = load_llm()

# # Step 3: Crawl arXiv
# def crawl_arxiv():
#     file_path = os.path.join(data_path, "arxiv_papers.txt")
#     open(file_path, "w").close()  # Clear existing data
    
#     class ArxivSpider(scrapy.Spider):
#         name = "arxiv_spider"
#         start_urls = ["https://arxiv.org/"]

#         def parse(self, response):
#             for category in response.xpath("//a[contains(@href, '/list/')]/@href").getall():
#                 yield response.follow(category, self.parse_category)

#         def parse_category(self, response):
#             for paper in response.xpath("//dl/dt"):
#                 title = paper.xpath(".//a[contains(@title, 'Abstract')]/text()").get()
#                 abstract_link = response.urljoin(paper.xpath(".//a[contains(@title, 'Abstract')]/@href").get())
#                 pdf_link = abstract_link.replace("abs", "pdf")
                
#                 with open(file_path, "a", encoding="utf-8") as f:
#                     f.write(f"Title: {title}\nAbstract Link: {abstract_link}\nPDF Link: {pdf_link}\n\n")
    
#     process = CrawlerProcess(settings={"LOG_LEVEL": "ERROR"})
#     process.crawl(ArxivSpider)
#     process.start()
#     print("✅ All arXiv papers saved in data/arxiv_papers.txt")

# crawl_arxiv()

# # Step 4: Create ChromaDB Index
# def fetch_arxiv_papers():
#     file_path = os.path.join(data_path, "arxiv_papers.txt")
#     with open(file_path, "r", encoding="utf-8") as f:
#         raw_text = f.read()
    
#     documents = raw_text.split("\n\n")
#     metadata = []
#     for doc in documents:
#         lines = doc.split("\n")
#         title = lines[0].replace("Title: ", "") if lines else "Unknown Title"
#         pdf_link = lines[2].replace("PDF Link: ", "") if len(lines) > 2 else "No URL"
#         metadata.append({"title": title, "source": pdf_link})
    
#     return documents, metadata

# documents, metadata = fetch_arxiv_papers()

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def create_chroma_index(documents, metadata):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = [Document(page_content=text, metadata=meta) for text, meta in zip(documents, metadata)]
#     split_docs = text_splitter.split_documents(docs)
    
#     chroma_client = chromadb.PersistentClient(path=data_path)
#     vectorstore = Chroma.from_documents(split_docs, embeddings, client=chroma_client)
#     print("✅ ChromaDB index created!")
#     return vectorstore

# vectorstore = create_chroma_index(documents, metadata)

# # Step 5: Search ChromaDB
# def search_chroma(query, top_k=3):
#     chroma_client = chromadb.PersistentClient(path=data_path)
    
#     if not os.path.exists(data_path):
#         print("⚠️ ChromaDB index not found. Please run indexing first.")
#         return []
    
#     vectorstore = Chroma(embedding_function=embeddings, client=chroma_client)
#     retriever = vectorstore.as_retriever()
#     docs = retriever.get_relevant_documents(query, top_k=top_k)

#     print("\n📄 **Top Research Papers:**")
#     for doc in docs:
#         print(f"- {doc.metadata.get('title', 'Unknown Title')} ( [PDF]({doc.metadata.get('source', 'No URL')}))")
    
#     return docs
