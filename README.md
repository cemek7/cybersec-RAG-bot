
# 🔐 Cybersecurity RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) assistant tailored for cybersecurity researchers, penetration testers, and threat analysts. This assistant ingests documents and web content (PDFs, DOCX, Markdown, and URLs), embeds and indexes them, and answers questions with contextually accurate responses using local or cloud-hosted LLMs.

---

## 🚀 Features

- 🧠 **RAG Pipeline**: Combine document retrieval and remote LLMs for accurate answers.  
- 📄 **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.md`, and URLs.  
- 🌐 **Web & GitHub Integration**: Ingest cybersecurity content from websites and GitHub repos.  
- ⚙️ **Modular Design**: Easily extendable pipelines (feature, training, inference).  
- 🧩 **Embeddings + ChromaDB**: Semantic vector search with sentence-transformers.  
- 🤖 **Remote LLM Support**: Uses Colab-hosted LLMs or local models like TinyLlama.  
- 🛡️ **Cybersecurity-Focused**: Optimized prompts, sources, and pipeline for pentesting, CVEs, and security analysis.  
- 📦 **FastAPI Backend**: Clean API for `/ingest` and `/ask`.

---

## 📁 Project Structure

<pre>
cybersec-rag/
├── app/
│   ├── server/
│   │   └── main.py             # FastAPI app
│   ├── rag/
│   │   ├── rag_pipeline.py     # Main RAG logic
│   │   ├── prompt_engine.py    # Prompt formatting (customizable)
│   │   └── llm_colab_client.py # Remote LLM communication
│   ├── data/
│   │   ├── ingestion.py        # File & URL ingestion
│   │   └── utils.py            # Helpers for parsing & cleaning
│   └── schema/
│       └── schema.py           # Pydantic schemas
├── vector_store/               # ChromaDB storage
├── requirements.txt
└── README.md
</pre>

---

## 🧪 Quickstart

### 1. Clone the Repo

```bash
git clone https://github.com/cemek7/cybersec-RAG-bot.git
cd cybersec-rag
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Backend

```bash
uvicorn app.server.main:app --reload
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the API via Swagger UI.

---

## 📤 Endpoints

### POST /ingest

Upload files or URLs for ingestion and embedding.

```bash
curl -X POST http://localhost:8000/ingest \\
  -F "file=@sample_report.pdf"
```

### POST /ask

Ask questions about ingested content.

Example payload:

```json
{
  "query": "What is CVE-2023-24055 and how is it exploited?"
}
```

---

## 🧠 LLM Support

- TinyLlama 1.1B (CPU) – for low-latency refinement locally  
- Remote LLMs via Colab (e.g., Mistral/Mixtral) – for main generation

Configure your cloud LLM endpoint inside `llm_colab_client.py`.

---

## 🛠️ Customization

- 🔧 Plug in your own embedding models or vector DBs  
- 🔌 Add new sources (e.g., RSS feeds, CVE feeds, GitHub orgs)  
- 🧪 Customize prompts in `prompt_engine.py`  
- 🔍 Enable metadata filters for domain-specific answers

---

## 📦 Dependencies

- `fastapi`, `uvicorn`  
- `langchain`, `sentence-transformers`  
- `chromadb`  
- `PyMuPDF`, `python-docx`, `markdown`, `beautifulsoup4`  
- `llama-cpp-python` *(for local inference)*  
- `aiohttp`, `pydantic`, `tqdm`

---

## ✅ Roadmap

- [x] File + URL ingestion  
- [x] Local + remote LLM support  
- [x] Modular RAG pipeline  
- [ ] GitHub repo auto-ingestion  
- [ ] Streamlit-based frontend UI  
- [ ] API key and auth middleware  
- [ ] Auto-CVE feed integration from NVD, Shodan, ExploitDB

---

## 👨‍💻 Maintainer

**christopher ozougwu** – [yourwebsite.com](https://yourwebsite.com)  
Built with ❤️ for security researchers, red teamers, and AI hobbyists.

---

## 📄 License

MIT License. See the `LICENSE` file for full details.


