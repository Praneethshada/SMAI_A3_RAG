# SMAI Assignment 3: RTI Assistant — Government Services RAG Chatbot

**Theme T10.3 — RTI Filing Helper**  
**Course:** Statistical Methods in AI
**Team:** The3
**Team Members:** Shada Praneeth Reddy (2025204006), Nagam Chandrakanth Reddy (2025204003), Pittada Sai Harsha Vardhan (2025204021)

---

## 1. Introduction & Problem Statement

Indian government services — particularly the Right to Information (RTI) Act — are backed by extensive official PDF documentation. These documents cover the legal framework, application procedures, fees, timelines, exemptions, and appeal mechanisms. For a typical citizen, navigating multiple dense PDFs to answer a specific procedural question is impractical.

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about RTI filing by retrieving relevant passages directly from the official RTI documents. Answers are grounded **only** in the retrieved context — the LLM cannot hallucinate or draw on outdated training knowledge. This approach requires no model training, makes every answer verifiable via source citations, and is immediately deployable.

---

## 2. Dataset Description

The knowledge base consists of five official documents downloaded from the Government of India's RTI portal:

| File | Content |
|---|---|
| `RTI Act 2005 (Amended)-English Version.pdf` | Full legal text of the RTI Act, 2005 |
| `RTI-Act.pdf` | Alternative copy of the RTI Act |
| `RTI_Rules_2019_hindi_english.pdf` | RTI Rules 2012 (amended 2019) — bilingual |
| `RTI FAQ English.pdf` | Frequently asked questions |
| `RTI Online User Manual English.pdf` | Step-by-step guide for the RTI Online Portal |

Together these provide comprehensive coverage: legal framework (Act), procedural details (Rules), quick answers (FAQ), and digital filing instructions (Manual). Each document is assigned a `doc_type` metadata tag during ingestion, enabling future filtered retrieval per document category.

---

## 3. System Architecture (RAG Pipeline)

The system is split into two independent phases.

### 3.1 Offline Ingestion Phase (`ingest.py`)

```
RTI PDFs  ──►  PyPDFLoader  ──►  SmartChunker  ──►  MiniLM Encoder  ──►  ChromaDB
(data/)         (text +          (700/150,           (384-dim dense       (persisted
                 metadata)        section-aware,       embeddings,          vector
                                  chunk_id,            local CPU)           index)
                                  doc_type)
```

**Steps:**

1. **Document Loading**: `PyPDFDirectoryLoader` extracts text and metadata (`source`, `page`) from every PDF in `data/`.

2. **Smart Chunking**: `RecursiveCharacterTextSplitter` is configured with RTI-specific separators that respect legal section boundaries (`Section N`, `Chapter IV`, `Rule N`) before falling back to paragraph → sentence → character splits. Chunk size is 700 characters with 150-character overlap.
   - *Why 700?*: `all-MiniLM-L6-v2` has a 512-token window (~384 words). 700 characters is approximately 140 words — well within the window for English legal text, ensuring no truncation during embedding.
   - *Why smaller than 1000?*: Finer chunks → more precise retrieval for specific facts (fees, deadlines, section numbers).

3. **Metadata Enrichment**:
   - `doc_type`: "Act", "Rules", "FAQ", or "Manual" — classified from filename.
   - `chunk_id`: `sha256(source + page + start_index)[:12]` — a deterministic 12-character hash. This makes ingestion **idempotent**: re-running `ingest.py` with the same PDFs never creates duplicate chunks in ChromaDB.

4. **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` runs locally on CPU. It produces 384-dimensional dense embeddings — a proven, lightweight model for semantic search on English text, with no API cost or latency.

5. **Vector Store**: ChromaDB persists embeddings to disk (`chroma_db/`). The cosine similarity index enables sub-second approximate nearest-neighbour search over thousands of chunks.

### 3.2 Online Query Phase (`app.py`)

```
User Query
    │
    ▼
History-Aware Condenser          ← chat_history + current Q
    │  (produces standalone Q)
    ▼
MMR Retriever (ChromaDB)         ← fetch_k=20, k=6, λ=0.7
    │  (6 diverse, relevant chunks)
    ▼
Prompt Builder                   ← system + context + history + query
    │  (+ Hindi flag if toggled)
    ▼
Groq LLM (llama-3.3-70b)        ← streaming=True, temp=0
    │  (streamed tokens)
    ▼
Streamlit UI                     ← answer + expandable citations
```

**Steps:**

1. **History-Aware Condenser** (`create_history_aware_retriever`): Before retrieval, the last N chat messages and the current question are sent to the LLM with the instruction: *"Rephrase this as a self-contained search query."* This resolves pronouns and implicit references in follow-up questions. Example: *"What about the fees?"* becomes *"What is the fee for filing an RTI application?"* — drastically improving retrieval precision.

2. **MMR Retrieval**: Instead of plain top-k similarity search, **Maximal Marginal Relevance** selects chunks that are both relevant *and* diverse. From a candidate pool of 20 chunks (`fetch_k=20`), MMR picks 6 (`k=6`) that maximise relevance while minimising redundancy (controlled by `λ=0.7`). This prevents the context window from being dominated by near-identical excerpts from the same section.

3. **Prompt Engineering**: The system prompt establishes the LLM as an RTI expert, provides the retrieved context, and — when the Hindi toggle is on — instructs the model to respond entirely in Hindi (Devanagari script). `temperature=0` ensures deterministic, factual answers appropriate for legal information.

4. **Streaming Response**: The LLM response is streamed token-by-token to the UI. This is critical for government-facing applications: users see progress immediately rather than waiting for a complete response (reducing perceived latency significantly).

5. **Source Citations**: Source documents are rendered as expandable UI elements showing the PDF filename, document type, and exact page numbers.

---

## 4. Implementation Details & ML Best Practices

### 4.1 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Frontend | Streamlit ≥ 1.32 | Rapid prototyping; built-in chat UI primitives |
| Document Parsing | PyPDF | Reliable PDF text extraction |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong semantic similarity |
| Vector Store | ChromaDB | Lightweight, no server required, cosine search |
| Retrieval | LangChain (MMR) | High-level abstractions + MMR support |
| LLM | Groq — llama-3.3-70b-versatile | Fast inference, free tier, high quality |
| Retry | Tenacity | Production-grade retry with exponential backoff |
| Orchestration | LangChain LCEL | Composable, testable pipeline components |

### 4.2 Embedding Model Choice

`all-MiniLM-L6-v2` is selected over larger alternatives because:
- It runs entirely locally (no API cost, no network dependency during ingestion)
- 384-dim embeddings are sufficient for lexically specific RTI queries (fees, sections, deadlines)
- It is the standard benchmark model for semantic search in the sentence-transformers ecosystem

### 4.3 Chunking Strategy

A naive fixed-size chunker risks splitting mid-sentence or mid-section, creating incoherent chunks. The custom separator list:
```
Section N / Chapter IV / Rule N  →  paragraph  →  sentence  →  word  →  character
```
ensures the splitter always tries to break at legal section boundaries first. `add_start_index=True` records the character offset within the source document, enabling the deterministic chunk ID.

### 4.4 Retrieval Quality: MMR vs. Top-K

Standard top-k retrieval can return highly similar chunks from the same paragraph. For a question like *"What are the exemptions?"* (Section 8, RTI Act), top-4 similarity might return four nearly-identical excerpts of Section 8, wasting the context window. MMR explicitly penalises chunks similar to already-selected ones, ensuring the context contains diverse perspectives — e.g., the Act text, the Rules expansion, and an FAQ summary.

### 4.5 Multi-turn Conversation Handling

The history-aware condenser is the most important ML design decision for usability. Without it, the system operates in zero-shot mode per query: each retrieval is blind to prior context. With it, the system correctly resolves:
- *"What is the RTI fee?"* → *"How can I pay it?"* → *"And what if I can't pay online?"*
A naive implementation would fail at the second and third questions; the condenser correctly generates *"RTI fee payment methods online"* and *"RTI fee offline payment or exemption"* as standalone queries.

---

## 5. System Design for Public/Government Use

### 5.1 Reliability
- **Tenacity retry**: 3 attempts with exponential backoff (2s → 4s → 8s) on Groq 429 rate-limit errors. The user sees a friendly message rather than a stack trace.
- **Graceful degradation**: If `chroma_db/` is missing, the app shows setup instructions instead of crashing.
- **Sanitized input**: User input is stripped and truncated to 1000 characters before being sent to the LLM.

### 5.2 Scalability
- **Stateless design**: All user session data is stored in `st.session_state` (client-side). The server holds no per-user state → multiple Streamlit workers can serve requests without shared state concerns.
- **`@st.cache_resource`**: The embedding model, ChromaDB connection, and LLM client are initialised once per worker process and reused for all subsequent requests — not recreated on every page interaction.

### 5.3 Usability
- **Hindi Language Toggle**: The most significant accessibility feature. Citizens who are more comfortable in Hindi can toggle the response language. The LLM responds in Devanagari script. The user can ask in English or Hindi.
- **Welcome message** with example questions reduces friction for first-time users.
- **Progressive streaming** gives immediate feedback; users don't stare at a blank screen.
- **Expandable citations** keep the chat clean while making sources verifiable on demand.
- **Clear Chat** button lets users start fresh without reloading the page.

---

## 6. Optional Feature: Hindi Language Toggle

The assignment specification lists *"Talk in Hindi toggle"* as an optional feature. This is fully implemented:

- A sidebar toggle **"🇮🇳 Respond in Hindi"** persists in `st.session_state.hindi_mode`.
- When active, the system prompt includes:
  > *"IMPORTANT: You MUST respond entirely in Hindi (Devanagari script). Even if the question is asked in English, answer in Hindi."*
- This leverages `llama-3.3-70b-versatile`'s strong multilingual capability.
- A confirmation message in Hindi is displayed in the sidebar when the mode is active.

---

## 7. Sample Interactions & Evaluation

**Query 1 (English):**  
*"What is the fee for filing an RTI application?"*  
**Response:** *"A request for obtaining information under sub-section (1) of Section 6 shall be accompanied by an application fee of rupees ten..."*  
**Source:** RTI_Rules_2019_hindi_english.pdf [Rules] — page 3

**Query 2 (Multi-turn follow-up):**  
*"And what if I belong to the BPL category?"*  
**Response:** *"Persons below the poverty line (BPL) are exempted from paying the application fee..."*  
**Source:** RTI_Rules_2019_hindi_english.pdf [Rules] — page 4

**Query 3 (Hindi mode active, English input):**  
*"What is the time limit for a PIO to respond?"*  
**Response:** *"सूचना का अधिकार अधिनियम 2005 की धारा 7(1) के अनुसार, जन सूचना अधिकारी को आवेदन प्राप्त होने के 30 दिनों के भीतर जानकारी प्रदान करनी होगी..."*  
**Source:** RTI Act 2005 [Act] — page 5

The source citations demonstrate that the system correctly identifies and attributes information to its origin document, building user trust.

---

## 8. Conclusion

This project delivers a robust, production-quality RAG chatbot for RTI filing assistance. Key accomplishments beyond the baseline:

1. **Hindi toggle** (optional assignment feature) — fully implemented
2. **Smart section-aware chunking** with deterministic IDs — ML best practice
3. **MMR retrieval** — improved context diversity over naive top-k
4. **History-aware multi-turn** — essential for real conversational usability
5. **Streaming + retry** — production-grade reliability and UX
6. **Stateless, cached architecture** — scalable system design for public use

The pipeline requires no model training and is grounded entirely in verified official documents, making it suitable for deployment as a public citizen assistance tool.

---

## References

1. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
2. RTI Act 2005, Government of India. https://rti.gov.in
3. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
4. LangChain Documentation. https://python.langchain.com
5. Groq API Documentation. https://console.groq.com/docs
