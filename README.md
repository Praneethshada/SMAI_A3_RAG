# RTI Assistant — Government Services RAG Chatbot

> **SMAI Assignment 3 · Theme T10.3 — RTI Filing Helper**  
> A production-quality Retrieval-Augmented Generation (RAG) chatbot for Indian RTI documentation.

---

## What It Does

Citizens can ask natural-language questions about filing a Right to Information (RTI) application in India. The system retrieves relevant passages from **official government PDFs** and generates concise, grounded answers — no hallucination, every answer is backed by a source citation.

---

## Architecture

### Text Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   OFFLINE: INGEST PHASE  (ingest.py)            │
│                                                                 │
│  RTI PDFs  ──►  PyPDFLoader  ──►  SmartChunker  ──►  MiniLM   │
│  (data/)           │          (700/150 chars,     Embeddings   │
│                    │           section-aware,        (local)   │
│                    │           chunk_id, doc_type)      │      │
│                    └────────────────────────────►  ChromaDB   │
│                                                   (chroma_db/) │
└─────────────────────────────────────────────────────────────────┘
                              │  persisted on disk
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ONLINE: QUERY PHASE  (app.py)                   │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  History-Aware Condenser                                   │ │
│  │  [chat history + query → standalone question]              │ │
│  └───────────────────────┬────────────────────────────────────┘ │
│                          │ condensed query                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  MMR Retriever  (fetch_k=20 → k=6, λ=0.7)                 │ │
│  │  Diversity-aware cosine similarity search in ChromaDB      │ │
│  └───────────────────────┬────────────────────────────────────┘ │
│                          │ 6 diverse, relevant chunks           │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Prompt Builder                                            │ │
│  │  system: RTI expert + [Hindi flag]                         │ │
│  │  context: retrieved chunks                                 │ │
│  │  human: condensed query                                    │ │
│  └───────────────────────┬────────────────────────────────────┘ │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Groq LLM  (llama-3.3-70b-versatile)                      │ │
│  │  streaming=True · temp=0 · tenacity retry on 429          │ │
│  └───────────────────────┬────────────────────────────────────┘ │
│                          │ answer + source docs                 │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Streamlit UI                                              │ │
│  │  • Hindi toggle in sidebar                                 │ │
│  │  • Expandable PDF source citations                         │ │
│  │  • Session state chat history (stateless server)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Diagram

See [`architecture.puml`](./architecture.puml) for the full PlantUML component + sequence diagram.  
Render it at [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/) or with any PlantUML plugin.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit ≥ 1.32 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local CPU) |
| Vector Store | ChromaDB (persisted on disk) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Orchestration | LangChain (retrieval chain + history-aware retriever) |
| Retry | Tenacity (exponential backoff) |
| Document Parsing | PyPDF |

---

## Key Features

| Feature | Description |
|---|---|
| 🇮🇳 **Hindi Language Toggle** | Sidebar switch — LLM responds in Hindi (Devanagari) |
| 💬 **Multi-turn Conversation** | History-aware condenser turns follow-up questions into standalone queries |
| 🔍 **MMR Retrieval** | Maximal Marginal Relevance — diverse, non-redundant context chunks |
| ⚡ **Streaming** | LLM tokens stream to UI progressively (low perceived latency) |
| 🔄 **Retry on Rate Limit** | Tenacity exponential backoff (3 attempts, 2–10 s) on Groq 429 errors |
| 📄 **Source Citations** | Expandable expanders show exact PDF filename and page numbers |
| 🆔 **Idempotent Ingestion** | Deterministic chunk IDs — re-running `ingest.py` never duplicates chunks |
| 📦 **doc_type Metadata** | Each chunk tagged as Act / Rules / FAQ / Manual — enables filtered retrieval |
| 🗂️ **Cached Pipeline** | `@st.cache_resource` — embeddings + DB + LLM loaded once per process |

---

## Repository Structure

```
.
├── app.py               # Streamlit chatbot (query pipeline)
├── ingest.py            # PDF ingestion + chunking + embedding pipeline
├── architecture.puml    # PlantUML component + sequence diagram
├── requirements.txt     # Python dependencies
├── report.md            # Assignment report
├── .env                 # Environment variables (GROQ_API_KEY, GROQ_MODEL)
├── data/                # Official RTI PDF documents
│   ├── RTI Act 2005 (Amended)-English Version.pdf
│   ├── RTI FAQ English.pdf
│   ├── RTI Online User Manual English.pdf
│   ├── RTI-Act.pdf
│   └── RTI_Rules_2019_hindi_english.pdf
└── chroma_db/           # ChromaDB vector index (auto-generated)
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create / edit `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Ingest Documents

```bash
# First run (build from scratch)
python ingest.py

# Re-run after adding new PDFs (idempotent — won't duplicate existing chunks)
python ingest.py

# Full reset (wipe and rebuild)
python ingest.py --reset
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Sample Queries

| Query | Expected Source |
|---|---|
| "What is the fee for filing an RTI?" | RTI_Rules_2019 |
| "How do I file an RTI online?" | RTI Online User Manual |
| "Who is a Public Information Officer?" | RTI Act 2005 |
| "What are the exemptions from disclosure?" | RTI Act 2005, Section 8 |
| "What happens if I don't get a response in 30 days?" | RTI Act / FAQ |

---

## ML & System Design Notes

- **Chunk size 700/150**: Tuned for MiniLM's 512-token window. Smaller than 1000 → more precise recall for specific facts.
- **MMR over top-k**: Prevents the retriever from returning 6 nearly-identical chunks from the same section.
- **History-aware condenser**: Essential for multi-turn usability — without it, follow-ups like *"What about appeals?"* would return irrelevant chunks.
- **Stateless session**: All state in `st.session_state` → no server-side user sessions → horizontally scalable.
- **Temperature = 0**: Deterministic, factual answers — appropriate for legal/government information.
