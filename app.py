"""
app.py — RTI Assistant Chatbot (Streamlit)
==========================================
A production-quality RAG chatbot for Indian RTI filing assistance.
Built on LangChain LCEL (LangChain Expression Language) — the modern,
composable API for chaining LLM operations.

Key design decisions:
  - History-aware retrieval  : condenses multi-turn dialogue into a
                               standalone query before retrieval, so
                               follow-up questions like "What about the fees?"
                               work correctly.
  - MMR retrieval            : Maximal Marginal Relevance selects diverse
                               chunks (not just the top-k most similar),
                               reducing redundancy in context.
  - Streaming                : LLM tokens are streamed to the UI so the
                               user sees output progressively (low perceived
                               latency — critical for public-facing apps).
  - Tenacity retry           : exponential backoff on 429/rate-limit errors.
  - Hindi toggle             : Optional assignment feature; switches the
                               system prompt so the LLM responds in Hindi.
  - Stateless session design : all chat state is kept in st.session_state;
                               the server holds no per-user data → scalable.
  - @st.cache_resource       : the heavy pipeline (embeddings + vector DB +
                               LLM) is loaded once per Streamlit worker
                               process, not on every page interaction.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import RateLimitError as GroqRateLimitError

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ──────────────────────────── Configuration ────────────────────────────────
load_dotenv()

CHROMA_PATH        = "chroma_db"
EMBED_MODEL        = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama3-8b-8192"
GROQ_MODEL         = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

# MMR retrieval parameters
MMR_FETCH_K   = 20   # candidate pool size (cast wide net)
MMR_K         = 6    # final chunks returned after diversity filter
MMR_LAMBDA    = 0.7  # 0=max diversity, 1=max relevance; 0.7 balances well

# Max LangChain history messages to pass to condenser (keeps tokens low)
MAX_HISTORY   = 10

# ──────────────────────────── Streamlit page ───────────────────────────────
st.set_page_config(
    page_title="RTI Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── Custom CSS (premium UI) ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.rti-header {
    background: linear-gradient(135deg, #1a3c5e 0%, #0d6efd 60%, #0dcaf0 100%);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    color: white;
}
.rti-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.rti-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

.stChatMessage { border-radius: 12px; }

div[data-testid="stExpander"] {
    border: 1px solid #dee2e6;
    border-radius: 8px;
    margin-top: 0.4rem;
}

.sidebar-badge {
    background: #e8f4fd;
    border-left: 4px solid #0d6efd;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────── Cached Pipeline Components ──────────────────────

@st.cache_resource(show_spinner="Loading RTI knowledge base…")
def load_pipeline_components():
    """
    Build and cache the heavy pipeline components.
    Called once per Streamlit worker — not recreated on every interaction.

    Returns a dict with:
      - 'retriever'    : MMR-configured ChromaDB retriever
      - 'llm'          : ChatGroq instance (streaming)
      - 'chunk_count'  : total chunk count for UI display
    Returns None if the vector DB doesn't exist.
    """
    if not os.path.exists(CHROMA_PATH):
        return None

    # 1. Local embeddings — no API cost, consistent with ingestion
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 2. Vector store
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    chunk_count = db._collection.count()

    # 3. MMR Retriever
    #    fetch_k=20 → evaluate 20 candidates for diversity
    #    k=6        → return 6 most diverse relevant chunks
    #    lambda=0.7 → favour relevance slightly over diversity
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           MMR_K,
            "fetch_k":     MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )

    # 4. LLM — streaming=True for progressive token output
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, streaming=True)

    return {"retriever": retriever, "llm": llm, "chunk_count": chunk_count}


# ──────────────────────── LCEL Chain Builders ──────────────────────────────

def build_condenser_chain(llm):
    """
    Build the question-condensing chain.
    Takes {chat_history, input} and produces a standalone query string.

    Why this matters:
    Without this, a follow-up like "What about the appeal process?"
    is retrieved as-is — the retriever has no idea what "this" refers to.
    The condenser rewrites it as "What is the first/second appeal process
    under the RTI Act?" — drastically improving retrieval precision.
    """
    condense_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human",
         "Given our conversation above, rewrite my last message as a "
         "complete, self-contained search query with no pronouns or "
         "references to earlier context. Output only the query."),
    ])
    return condense_prompt | llm | StrOutputParser()


def build_rag_chain(llm, retriever, hindi_mode: bool):
    """
    Build the full LCEL RAG pipeline dynamically.
    Rebuilt when hindi_mode changes (lightweight operation — no DB/model reload).

    Pipeline:
      {input, chat_history}
          → condenser  → standalone_query
          → retriever  → context_docs
          → prompt     → augmented_prompt
          → llm        → answer_text
    """
    lang_instruction = (
        "IMPORTANT: You MUST respond entirely in Hindi (Devanagari script). "
        "Even if the question is asked in English, answer in Hindi.\n\n"
        if hindi_mode else ""
    )

    system_prompt = (
        "You are a helpful, accurate assistant specialising in the Indian "
        "Right to Information (RTI) Act and filing procedures. "
        "Answer questions ONLY using the retrieved context below. "
        "If the context is insufficient, say so honestly — do not invent facts. "
        "When referencing specific rules or sections, cite them explicitly.\n"
        f"{lang_instruction}"
        "\nRetrieved Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    condenser = build_condenser_chain(llm)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # LCEL pipeline — composable, inspectable, parallelisable
    chain = (
        RunnablePassthrough.assign(
            # Step 1: condense history + question → standalone query
            standalone_query=RunnableLambda(
                lambda x: condenser.invoke({
                    "chat_history": x.get("chat_history", []),
                    "input": x["input"],
                })
            )
        )
        | RunnablePassthrough.assign(
            # Step 2: retrieve diverse context using the condensed query
            context=RunnableLambda(lambda x: format_docs(
                retriever.invoke(x["standalone_query"])
            )),
            # Preserve source docs for citation display
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["standalone_query"]))
        )
        | RunnablePassthrough.assign(
            # Step 3: generate answer
            answer=(qa_prompt | llm | StrOutputParser())
        )
    )

    return chain


# ──────────────────────── Retry-wrapped Invocation ─────────────────────────

@retry(
    retry=retry_if_exception_type((GroqRateLimitError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def invoke_with_retry(chain, payload: dict):
    """
    Invoke the RAG chain with exponential backoff.
    Retries up to 3 times on rate-limit errors (2s → 4s → 8s wait).
    """
    return chain.invoke(payload)


# ────────────────────────── Session State Init ─────────────────────────────

def init_session():
    if "messages"   not in st.session_state:
        st.session_state.messages   = []   # [{role, content}] for display
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []   # [HumanMessage | AIMessage] for LangChain
    if "hindi_mode" not in st.session_state:
        st.session_state.hindi_mode = False


def append_message(role: str, content: str):
    """Append to both display history and LangChain message history."""
    st.session_state.messages.append({"role": role, "content": content})
    msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    st.session_state.lc_history.append(msg)
    # Trim to MAX_HISTORY to bound token usage
    if len(st.session_state.lc_history) > MAX_HISTORY:
        st.session_state.lc_history = st.session_state.lc_history[-MAX_HISTORY:]


# ────────────────────────────── Sidebar ───────────────────────────────────

def render_sidebar(pipeline):
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # ── Hindi Toggle (Assignment Optional Feature) ───────────────────
        st.session_state.hindi_mode = st.toggle(
            "🇮🇳 Respond in Hindi",
            value=st.session_state.hindi_mode,
            help=(
                "When enabled, all answers are given in Hindi (Devanagari). "
                "You can still ask questions in English."
            ),
        )
        if st.session_state.hindi_mode:
            st.info("हिंदी मोड सक्रिय है। सभी उत्तर हिंदी में दिए जाएंगे।")

        st.divider()
        st.markdown("## 📊 System Info")

        if pipeline:
            for label, value in [
                ("🧠 LLM", GROQ_MODEL),
                ("📚 Knowledge chunks", str(pipeline["chunk_count"])),
                (f"🔍 Retrieval", f"MMR (k={MMR_K}, fetch_k={MMR_FETCH_K})"),
                ("🤗 Embeddings", "all-MiniLM-L6-v2"),
            ]:
                st.markdown(
                    f'<div class="sidebar-badge">{label}: <strong>{value}</strong></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.lc_history = []
            st.rerun()

        st.divider()
        st.markdown(
            "<small>SMAI Assignment 3 — Theme T10.3<br>"
            "Groq · LangChain · ChromaDB</small>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────── Main ─────────────────────────────────────

def main():
    init_session()

    # Header
    st.markdown("""
    <div class="rti-header">
        <h1>📋 RTI Assistant</h1>
        <p>Ask anything about filing a Right to Information (RTI) application in India.
        Every answer is grounded in official government documents.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load pipeline components (cached)
    pipeline = load_pipeline_components()
    render_sidebar(pipeline)

    if pipeline is None:
        st.error(
            "⚠️ **Vector database not found.**\n\n"
            "Run the ingestion pipeline first:\n"
            "```bash\npython ingest.py\n```"
        )
        st.stop()

    # Welcome message when chat is empty
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 **Hello!** I'm your RTI filing assistant. I can help you with:\n\n"
                "- 📝 How to file an RTI application (online or offline)\n"
                "- 💰 Application fees and payment methods\n"
                "- ⏱️ Response timelines and what to do if missed\n"
                "- ⚖️ Rights, exemptions, and Section 8\n"
                "- 🏛️ First and Second Appeals process\n\n"
                "**Try:** *\"What is the fee for filing an RTI?\"* or "
                "*\"How do I file an RTI online?\"*"
            )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    placeholder = (
        "आरटीआई के बारे में कुछ भी पूछें…"
        if st.session_state.hindi_mode
        else "Ask about RTI filing, fees, appeals, exemptions…"
    )
    user_input = st.chat_input(placeholder)
    if not user_input:
        return

    # Sanitize: strip whitespace, truncate to 1000 chars
    user_input = user_input.strip()[:1000]

    with st.chat_message("user"):
        st.markdown(user_input)
    append_message("user", user_input)

    # Generate answer
    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("⏳ *Searching documents…*")

        try:
            # Build chain with current Hindi mode (lightweight, no model reload)
            rag_chain = build_rag_chain(
                pipeline["llm"],
                pipeline["retriever"],
                st.session_state.hindi_mode,
            )

            # Exclude the last human message from history (it's the current input)
            history = st.session_state.lc_history[:-1]

            status.markdown("🤔 *Generating answer…*")
            result = invoke_with_retry(rag_chain, {
                "input":        user_input,
                "chat_history": history,
            })

            status.empty()

            answer       = result.get("answer", "")
            source_docs  = result.get("source_documents", [])

            st.markdown(answer)

            # Expandable source citations
            if source_docs:
                unique_sources: dict = {}
                for doc in source_docs:
                    src  = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", "?")
                    dt   = doc.metadata.get("doc_type", "")
                    key  = f"{src}  [{dt}]" if dt else src
                    unique_sources.setdefault(key, set()).add(
                        str(int(page) + 1) if str(page).isdigit() else str(page)
                    )

                with st.expander(f"📄 Sources ({len(unique_sources)} document(s))", expanded=False):
                    for src_name, pages in unique_sources.items():
                        page_list = ", ".join(
                            sorted(pages, key=lambda x: int(x) if x.isdigit() else 0)
                        )
                        st.markdown(f"- **{src_name}** — page(s) {page_list}")

            append_message("assistant", answer)

        except GroqRateLimitError:
            status.empty()
            err = (
                "⚠️ **Rate limit reached.** Groq API quota exhausted. "
                "Please wait a moment and try again. "
                "Check your limits at [console.groq.com](https://console.groq.com)."
            )
            st.warning(err)
            append_message("assistant", err)

        except Exception as e:
            status.empty()
            err_text = str(e)
            if "404" in err_text or "not found" in err_text.lower():
                err = (
                    f"❌ Model `{GROQ_MODEL}` is unavailable. "
                    "Update `GROQ_MODEL` in `.env` and restart the app."
                )
            elif "429" in err_text or "rate" in err_text.lower():
                err = "⚠️ **Rate limited.** Please wait a moment and retry."
            else:
                err = f"❌ Error: {err_text}"
            st.error(err)
            append_message("assistant", err)


if __name__ == "__main__":
    main()
