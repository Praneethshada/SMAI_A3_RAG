# Technical Report: RTI Assistant — A RAG-Based Chatbot for Indian Government RTI Services

**Course:** Statistical Methods in AI (SMAI), Assignment 3, Theme T10.3

**Team Name:** The3

**Team Members:** 
1. Shada Praneeth Reddy 2025204006 (shadapraneeth.reddy@students.iiit.ac.in)
2. Nagam Chandrakanth Reddy 2025204003 (nagam.reddy@students.iiit.ac.in)
3. Pittada Sai Harsha Vardhan 2025204021 (pittada.s@students.iiit.ac.in)

**GitHub:** [https://github.com/Praneethshada/SMAI_A3_RAG](https://github.com/Praneethshada/SMAI_A3_RAG)

**Demo Video (Drive):** [https://drive.google.com/file/d/17KivbFgwDYT864f6cWiiiWzeE496aVZl/view?usp=sharing](https://drive.google.com/file/d/17KivbFgwDYT864f6cWiiiWzeE496aVZl/view?usp=sharing)

**Demo Video (YouTube):** [https://youtu.be/i-qibG_g0_c](https://youtu.be/i-qibG_g0_c)


## 1. Introduction

Indian government services, particularly the Right to Information (RTI) Act, are backed by extensive official PDF documentation covering legal definitions, procedural rules, online filing steps, and FAQs. For a typical citizen, locating a specific fact such as the application fee, response deadline, or appeal process requires manually scanning multiple dense documents, which is impractical.

We address this with a Retrieval-Augmented Generation (RAG) chatbot that:

1. Indexes official RTI PDFs into a local vector database at ingestion time.
2. At query time, retrieves the most semantically relevant passages and supplies them as grounded context to a large language model (LLM).
3. Generates a concise, verifiable answer with source citations, referencing only retrieved passages and not the model's parametric knowledge.

This design eliminates hallucination risk, makes every answer auditable, and requires no model training or fine-tuning. The system is packaged as a Streamlit web application.

![Home Page](App%20Screenshots/Home%20Page.png)

*Figure 1: RTI Assistant home page*

![Home Page 2](App%20Screenshots/Home%20Page%202.png)

*Figure 2: Home page showing the topic cards.*


## 2. Data

### 2.1 Knowledge Base Documents

Five official documents sourced from the Government of India RTI Portal ([rti.gov.in](https://rti.gov.in)):

| File | Content | Type Tag |
|---|---|---|
| `RTI Act 2005 (Amended)-English Version.pdf` | Full text of the RTI Act, 2005 | Act |
| `RTI-Act.pdf` | Alternate copy of the RTI Act | Act |
| `RTI_Rules_2019_hindi_english.pdf` | RTI Rules 2012/2019, bilingual | Rules |
| `RTI FAQ English.pdf` | 27 FAQs covering common citizen queries | FAQ |
| `RTI Online User Manual English.pdf` | Step-by-step guide for the RTI Online Portal | Manual |

### 2.2 Corpus Statistics

After ingestion: 88 raw pages produce 539 text chunks (average 536 characters per chunk).

| Doc Type | Chunks |
|---|---|
| Act | ~280 |
| Manual | ~149 |
| Rules | ~65 |
| FAQ | ~45 |

### 2.3 Data Quality Notes

PDFs are parsed using PyPDF via LangChain's `PyPDFDirectoryLoader`. The bilingual Rules PDF retains Hindi text, which benefits Hindi-mode retrieval. One PDF (`RTI-Act.pdf`) triggers a non-standard header warning but is parsed correctly after fallback recovery by the library.


## 3. Method

### 3.1 Ingestion Pipeline (ingest.py)

The offline pipeline runs once and produces the persistent vector index.

```
RTI PDFs -> PyPDFLoader -> SmartChunker -> MetadataEnricher -> MiniLM Encoder -> ChromaDB
```

**Step 1: Document Loading.** `PyPDFDirectoryLoader` reads all `.pdf` files in `data/`, extracting page text and metadata (source filepath, page number).

**Step 2: Smart Chunking.** `RecursiveCharacterTextSplitter` uses a custom separator hierarchy that respects RTI legal structure before falling back to finer splits:

```
Section N / Chapter IV / Rule N  ->  paragraph  ->  line  ->  sentence  ->  word  ->  character
```

Parameters: `chunk_size=700`, `chunk_overlap=150`, `add_start_index=True`.

Why 700 characters? The `all-MiniLM-L6-v2` model has a 512-token context window, corresponding to approximately 700 to 900 English characters. Staying within this boundary avoids embedding truncation. Smaller chunks also improve precision: a question about the RTI fee retrieves the exact clause stating Rs. 10 rather than a large paragraph that merely mentions fees in passing.

**Step 3: Metadata Enrichment.** Each chunk receives two new fields:

- `doc_type`: Keyword-classified from filename (Act, Rules, FAQ, Manual). Enables future filtered retrieval per document category.
- `chunk_id`: `sha256(source_path + "::" + page + "::" + start_index)[:12]`, a deterministic 12-character hash. This makes ingestion idempotent: re-running after adding new PDFs only embeds new chunks, never duplicating existing ones.

**Step 4: Embedding and Persistence.** Chunks are embedded with `all-MiniLM-L6-v2` (384-dimensional, runs locally on CPU, no API cost). Vectors are stored in ChromaDB with automatic disk persistence.

### 3.2 Query Pipeline (app.py)

Built using LangChain LCEL (LangChain Expression Language), the modern composable API.

```
User Query
  -> History-Aware Condenser
  -> MMR Retriever (ChromaDB)
  -> Prompt Builder (with Hindi flag)
  -> Groq LLM (temperature=0)
  -> Streamlit UI (answer + expandable citations)
```

**History-Aware Question Condensing.**
Before retrieval, a dedicated LLM call rewrites the current user message into a self-contained search query incorporating the conversation history. This is implemented as an LCEL chain: `condense_prompt | groq_llm | StrOutputParser()`.

Without condensing, a follow-up question like "What about BPL applicants?" is retrieved verbatim. The retriever has no context of what was previously discussed and returns irrelevant results.

With condensing, the chain produces "RTI application fee exemption for BPL (below poverty line) applicants", which retrieves the correct exemption clause with high precision.

**MMR Retrieval.**
Maximal Marginal Relevance (MMR) retrieval prevents the context window from being flooded with near-duplicate passages from the same section. From a candidate pool of 20 chunks (`fetch_k=20`), MMR selects 6 (`k=6`) that maximise both relevance and diversity (`lambda_mult=0.7`).

Example for the query "What are the exemptions under RTI?":
- Top-6 cosine similarity: 5 near-identical Section 8 excerpts from one PDF plus 1 FAQ entry.
- MMR: 2 Section 8 passages (Act) plus Rules expansion plus FAQ summary plus Manual reference plus amended clause. This gives the LLM a richer, more complete context.

**Prompt Engineering.**
The system prompt instructs the LLM to act as an RTI domain expert, answer only from retrieved context, cite specific sections and rules, and conditionally respond in Hindi when the toggle is active. Temperature is set to 0 for deterministic, factual output, which is essential for legal and government information.

**Retry Logic.**
All LLM calls are wrapped with Tenacity exponential backoff: 3 attempts with waits of 2 seconds, 4 seconds, and 8 seconds on `GroqRateLimitError` (HTTP 429). This provides graceful degradation for a public-facing service.

**Streaming Output.**
The LLM client is initialised with `streaming=True`. The Streamlit UI renders status placeholders during retrieval and then the full answer via `st.markdown()` once generation completes. This reduces perceived latency: users see immediate feedback rather than a blank screen during LLM generation.

### 3.3 System Architecture

The Streamlit frontend has three views navigable from a persistent sidebar:

| View | Purpose |
|---|---|
| Home | Chat interface; hero card and topic cards when chat is empty |
| Resources | Browsable PDF list with in-browser download links |
| Settings | Language toggle and model configuration display |

**Key system design decisions for public and government usability:**

| Decision | Rationale |
|---|---|
| `@st.cache_resource` pipeline caching | Embeddings, DB, and LLM loaded once per worker; low latency under concurrent use |
| Stateless session design (all state in `st.session_state`) | Server holds no per-user data, horizontally scalable, no sticky sessions |
| Input sanitization (strip and truncate to 1000 chars) | Prevents prompt injection and runaway token usage |
| History window cap (MAX_HISTORY=10 messages) | Bounds token cost per query regardless of conversation length |
| Graceful DB-missing error page | Setup instructions shown instead of unhandled crash |

**Component Architecture Diagram:**

![Architecture Diagram](Architecture%20&%20Sequence%20Diagrams/Architecture%20Diagram.png)

*Figure 3: Component architecture diagram showing the offline ingestion pipeline and online query pipeline.*

**Query Sequence Diagram:**

![Sequence Diagram](Architecture%20&%20Sequence%20Diagrams/Sequence%20Diagram.png)

*Figure 4: Sequence diagram of a multi-turn query showing the condenser, MMR retriever, LLM call, and retry logic.*

### 3.4 Optional Feature: Hindi Language Toggle

The assignment specification lists "Talk in Hindi toggle" as an optional feature. This is fully implemented.

A sidebar toggle (mirrored in the Settings page) injects a language directive into the system prompt:
- Hindi ON: "You MUST respond entirely in Hindi (Devanagari script). Even if the question is in English, answer in Hindi."
- Hindi OFF: "Respond ONLY in English", which explicitly prevents language drift from prior Hindi context in the conversation history.

The toggle persists across the session. The chat input placeholder also switches to Hindi when the mode is active.


## 4. Results

### 4.1 Sample Interactions

The following screenshots show the system responding to representative RTI queries. Screenshots are taken from the local running application.

![English Question](App%20Screenshots/English%20Question.png)

*Figure 5: English mode question and response*

![Citations](App%20Screenshots/Citiations.png)

*Figure 6: Reference sources expander showing PDF name, document type, and page numbers.*


![Hindi Question](App%20Screenshots/Hindi%20Question.png)

*Figure 7: Hindi mode active, showing a Devanagari response to the query.*

### 4.2 Application Pages

![Resources](App%20Screenshots/Resources.png)

*Figure 8: Resources page listing all knowledge-base PDFs with download links.*

![Settings](App%20Screenshots/Settings.png)

*Figure 9: Settings page showing the language toggle and model configuration.*

### 4.3 Ablation: Effect of Chunk Size

| Chunk Size | Overlap | Chunks | Retrieval Quality |
|---|---|---|---|
| 1000 | 200 | ~380 | Returns large paragraphs; specific facts sometimes missed |
| 700 | 150 | 539 | Retrieves specific clauses, section numbers, exact figures; best precision |
| 400 | 100 | ~900 | Over-fragmented; splits mid-sentence; loses surrounding context |

Chunk size 700 is optimal for this domain: precise enough to isolate individual facts, large enough to preserve sentence-level context.

### 4.4 Retrieval Strategy Comparison

| Retrieval Method | Context Diversity | Answer Completeness |
|---|---|---|
| Top-k cosine (k=6) | Low, near-duplicates dominate | Misses complementary sections |
| MMR (fetch_k=20, k=6, lambda=0.7) | High, diverse document types | Complete, Act plus Rules plus FAQ |

**Demo Video:**

[Watch on YouTube](https://youtu.be/i-qibG_g0_c) — [Watch on Google Drive](https://drive.google.com/file/d/17KivbFgwDYT864f6cWiiiWzeE496aVZl/view?usp=sharing)


## 5. Limitations

1. **PDF Parsing Quality.** PyPDF extracts raw text without understanding layout. Tables such as fee schedules lose column alignment. Libraries like `pdfplumber` or `unstructured` would improve structured data extraction.

2. **Monolingual Embeddings.** `all-MiniLM-L6-v2` is English-dominant. Queries about content that appears primarily in the Hindi section of the bilingual Rules PDF may have reduced retrieval accuracy. A multilingual model such as `paraphrase-multilingual-MiniLM-L12-v2` would address this.

3. **Static Knowledge Base.** The vector store is built at ingestion time. RTI rule amendments require re-running `ingest.py` with updated PDFs. An automated document change detection pipeline would be needed for production deployment.

4. **API Rate Limits.** The free Groq tier has daily token limits. Under heavy concurrent load the retry mechanism is triggered. Production deployment would require a paid tier or API key rotation.


## 6. References

1. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. arXiv:2005.11401
2. Carbonell, J. and Goldstein, J. (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. SIGIR 1998.
3. Government of India. (2005). The Right to Information Act, 2005. [https://rti.gov.in](https://rti.gov.in).
4. Government of India. (2019). The Right to Information (Amendment) Rules, 2019. Ministry of Personnel, Public Grievances and Pensions.
