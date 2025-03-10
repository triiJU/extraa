Arxiv-RAG: a retrieval-augmented generation (RAG) system that leverages **Mistral-7B** (& LLaMA-2) to search and summarize research papers from **arXiv**. It uses **ChromaDB** for efficient storage and retrieval.

app_file: streamlitapp.py

short_description: ArXiv research retrieval system


Features:
 - Dynamic arXiv Paper Retrieval – Fetch research papers across multiple categories in real time.
 - AI-Powered Summaries – Leverages Mistral-7B / LLaMA-2 for intelligent summarization.
 - Vector-Based Search – Uses ChromaDB for fast and efficient research paper retrieval.
 - Interactive Web UI – Built with Streamlit for an intuitive search experience.
 - Deployment & Automation – Hosted on Railway.app with GitHub CI/CD integration.


# Installation

git clone https://github.com/triiJU/arxiv_search_retrieval.git  
cd arxiv_search_retrieval  
conda create -n arxiv-rag python=3.10 -y  
conda activate arxiv-rag  
pip install -r requirements.txt  


# Usage

### Fetch & Index Papers

python fetch_arxiv.py

### Start Streamlit UI

streamlit run streamlitapp.py

### Search & Generate Answers

python query_chroma.py


## Example Queries
|          Query         |          Expected Output       |
|------------------------|--------------------------------|
| "Transformer models"   |   List of papers + summaries   |
|"Reinforcement learning"| Papers + AI-generated insights |


## Troubleshooting
pip check 

pip install --force-reinstall -r requirements.txt  

rm -rf chroma_db/ && python fetch_arxiv.py  



## License
MIT License.
