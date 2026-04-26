# SMAI Assignment 3: Government-Services RAG Chatbot (RTI Assistant)

## 1. Introduction & Problem Statement
Indian government services, such as the Right to Information (RTI) filings, involve extensive public PDF documentation containing rules, FAQs, and application procedures. Citizens often find it challenging to navigate these dense documents to find specific answers like "How do I apply?", "What are the fees?", or "What are the exemptions?". 

This project aims to solve this problem by building a Retrieval-Augmented Generation (RAG) chatbot. Instead of relying on general LLM knowledge (which can hallucinate or be outdated), the chatbot retrieves relevant passages directly from the official RTI documentation and generates answers grounded *only* in those documents. 

## 2. Dataset Description
The dataset for this project consists of official documentation downloaded from the RTI portal (Variant T10.3). The files include:
- `RTI Act 2005 (Amended)-English Version.pdf`
- `RTI FAQ English.pdf`
- `RTI Online User Manual English.pdf`
- `RTI-Act.pdf`
- `RTI_Rules_2019_hindi_english.pdf`

These documents provide a comprehensive knowledge base covering the legal framework, procedural rules, online application steps, and frequently asked questions regarding the RTI process.

## 3. System Architecture (RAG Pipeline)
The system is built on a standard RAG pipeline comprising two main phases: Data Ingestion and Question Answering.

### 3.1 Data Ingestion Phase
1. **Document Loading:** LangChain's `PyPDFDirectoryLoader` reads all PDFs in the `data/` directory and extracts the text along with metadata (source filename and page number).
2. **Text Splitting:** To fit within the context window of the embedding model and the LLM, the text is split into chunks using the `RecursiveCharacterTextSplitter` (chunk size of 1000 characters, with an overlap of 200 characters to maintain context across chunk boundaries).
3. **Embeddings:** The chunks are passed through the `sentence-transformers/all-MiniLM-L6-v2` model (running locally on CPU) to generate dense vector embeddings. This model is lightweight and highly effective for semantic search.
4. **Vector Database:** The embeddings and their corresponding text chunks are persisted locally using `ChromaDB`, creating a searchable vector index.

### 3.2 Question Answering Phase
1. **Query Processing:** The user inputs a question via the Streamlit Chat UI.
2. **Retrieval:** The query is embedded using the same MiniLM model, and ChromaDB performs a similarity search to retrieve the top $K$ (e.g., $K=4$) most relevant text chunks.
3. **Augmented Generation:** The retrieved context is combined with the user query into a system prompt. This prompt is sent to Google's Gemini 1.5 Flash LLM.
4. **Response:** The LLM generates a concise answer based strictly on the provided context. The application then appends the source citations (PDF name and page numbers) to the answer before displaying it to the user.

## 4. Implementation Details & Tools Used
- **Streamlit:** Used for the frontend application to provide a simple, conversational chat interface.
- **LangChain:** The orchestration framework used to connect document loaders, text splitters, vector stores, and the LLM into a cohesive pipeline.
- **ChromaDB:** A lightweight, open-source vector database used for storing embeddings and performing fast similarity searches.
- **Google Gemini 1.5 Flash:** A fast, cost-effective, and highly capable LLM accessed via the free Google AI Studio API (`langchain-google-genai`). It handles the final reasoning and generation step.
- **Sentence Transformers:** Used via `langchain-huggingface` to compute embeddings locally without incurring API costs or latency.

## 5. Sample Interactions & Evaluation
During manual verification, the system successfully grounded its responses. For example:
- **Query:** "What is the fee for filing an RTI?"
- **Response:** "A request for obtaining information under sub-section (1) of section 6 shall be accompanied by an application fee of rupees ten..."
- **Citation:** The system accurately cites the `RTI_Rules_2019` or the `RTI FAQ English.pdf` as the source.

The inclusion of source citations builds trust, allowing users to verify the LLM's claims directly in the official documents.

## 6. Conclusion & Future Work
We successfully built a functional RAG chatbot that makes Indian RTI documentation highly accessible. The pipeline requires no model training, relying entirely on open-source vector search and a free LLM API. 

**Future improvements could include:**
1. Integrating an optional "Talk in Hindi" toggle by modifying the LLM prompt.
2. Handling tabular data within the PDFs more robustly using advanced parsing libraries.
3. Deploying the application to Hugging Face Spaces or Streamlit Community Cloud for public access.
