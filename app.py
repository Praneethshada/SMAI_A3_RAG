import os
import base64
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

load_dotenv()

CHROMA_PATH        = "chroma_db"
EMBED_MODEL        = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama3-8b-8192"
GROQ_MODEL         = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

MMR_FETCH_K = 20
MMR_K       = 6
MMR_LAMBDA  = 0.7
MAX_HISTORY = 10

# ───────────── Streamlit page ────────────────────
st.set_page_config(
    page_title="RTI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────── Iconography (SVG) ─────────────

def get_icon_svg(name, size=20, color="currentColor"):
    icons = {
        "scale": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
        "settings": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
        "database": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
        "cpu": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
        "search": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
        "trash": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>',
        "message": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
        "book": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>',
        "zap": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
        "clipboard": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/></svg>',
        "credit-card": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/></svg>',
        "clock": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "gavel": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m14 5 3-3 5 5-3 3-5-5Z"/><path d="m5 14 7-7"/><path d="m2 17 5 5"/><path d="M3 21h18"/></svg>',
        "help": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
        "file-text": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "download": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
        "user": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "bot": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>'
    }
    return icons.get(name, "")

def get_icon_data_uri(name, size=24, color="currentColor"):
    svg_str = get_icon_svg(name, size, color)
    b64 = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8').replace('\n', '')
    return f"data:image/svg+xml;base64,{b64}"

def render_icon(name, size=20, color="currentColor"):
    return f'<span style="vertical-align: middle; margin-right: 8px;">{get_icon_svg(name, size, color)}</span>'


# ─────────────────────── Custom CSS (Premium UI) ───────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {{
    --primary: #2563eb;
    --primary-dark: #1e40af;
    --primary-light: #eff6ff;
    --bg-light: #f8fafc;
    --border: #e2e8f0;
    --text-main: #1e293b;
    --text-muted: #64748b;
}}

html, body, [class*="css"] {{ 
    font-family: 'Inter', sans-serif; 
}}

/* Hide scrollbar for a cleaner app look */
::-webkit-scrollbar {{
    display: none;
}}
* {{
    -ms-overflow-style: none;
    scrollbar-width: none;
}}


h1, h2, h3, .brand-title {{
    font-family: 'Outfit', sans-serif;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: white !important;
    border-right: 1px solid var(--border);
}}

.sidebar-logo {{
    padding: 1.5rem 1rem;
    margin-bottom: 1rem;
}}
.sidebar-logo h2 {{ margin: 0; font-size: 1.4rem; color: var(--primary); font-weight: 700; }}
.sidebar-logo p {{ margin: 0; font-size: 0.85rem; color: var(--text-muted); }}

.nav-item {{
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-main);
    transition: all 0.2s;
}}
.nav-item:hover {{ background: var(--primary-light); color: var(--primary); }}
.nav-item.active {{ background: var(--primary-light); color: var(--primary); font-weight: 600; }}

/* Hero Card */
.hero-card {{
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-radius: 24px;
    padding: 3rem;
    color: white;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 25px -5px rgba(37, 99, 235, 0.2);
    position: relative;
    overflow: hidden;
}}
.hero-card h1 {{ font-size: 2.8rem; margin: 0; font-weight: 700; display: flex; align-items: center; gap: 15px; }}
.hero-card p {{ font-size: 1.2rem; margin-top: 1rem; opacity: 0.9; max-width: 600px; font-weight: 300; }}

/* Quick Access Grid */
.quick-access-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 3rem;
}}
.access-card {{
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid var(--border);
    transition: all 0.2s;
    cursor: pointer;
    display: flex;
    gap: 15px;
}}
.access-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    border-color: var(--primary);
}}
.icon-box {{
    background: var(--primary-light);
    color: var(--primary);
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}
.card-content h4 {{ margin: 0; font-size: 1.1rem; color: var(--text-main); }}
.card-content p {{ margin: 5px 0 0; font-size: 0.9rem; color: var(--text-muted); line-height: 1.4; }}

/* Chat Interface */
.stChatMessage {{ 
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 1.2rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}}
.stChatMessage[data-testid="stChatMessageUser"] {{ background: var(--primary-light) !important; }}

/* Sidebar Badge */
.sidebar-badge {{
    background: #f1f5f9;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
}}
.sidebar-badge strong {{ color: var(--primary); }}

/* Resource List */
.resource-card {{
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 15px;
}}
.resource-card:hover {{ border-color: var(--primary); background: var(--primary-light); }}

</style>
""", unsafe_allow_html=True)






# ──────────────────────── Cached Pipeline Components ──────────────────────

@st.cache_resource(show_spinner="Loading RTI knowledge base…")
def load_pipeline_components():
    """Load embeddings, vector store, and LLM once per worker process."""
    if not os.path.exists(CHROMA_PATH):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    chunk_count = db._collection.count()

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA},
    )
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, streaming=True)

    return {"retriever": retriever, "llm": llm, "chunk_count": chunk_count}


# ──────────────────────── LCEL Chain Builders ──────────────────────────────

def build_condenser_chain(llm):
    """Rewrite the latest user message into a standalone search query."""
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
    """Build the LCEL RAG pipeline. Rebuilt on hindi_mode change (no model reload)."""
    if hindi_mode:
        lang_instruction = (
            "IMPORTANT: You MUST respond entirely in Hindi (Devanagari script). "
            "Even if the question is asked in English, answer in Hindi.\n\n"
        )
    else:
        lang_instruction = (
            "IMPORTANT: You MUST respond entirely in English. "
            "Even if the chat history or question contains Hindi, ignore it and reply ONLY in English.\n\n"
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

    chain = (
        RunnablePassthrough.assign(
            standalone_query=RunnableLambda(
                lambda x: condenser.invoke({
                    "chat_history": x.get("chat_history", []),
                    "input": x["input"],
                })
            )
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(
                retriever.invoke(x["standalone_query"])
            )),
            source_documents=RunnableLambda(
                lambda x: retriever.invoke(x["standalone_query"])
            )
        )
        | RunnablePassthrough.assign(
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
    """Invoke chain with exponential backoff (3 attempts, 2s/4s/8s)."""
    return chain.invoke(payload)


# ────────────────────────── Session State Init ─────────────────────────────

def init_session():
    if "messages"   not in st.session_state:
        st.session_state.messages   = []
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []
    if "hindi_mode" not in st.session_state:
        st.session_state.hindi_mode = False
    if "view"       not in st.session_state:
        st.session_state.view       = "Home"

def set_view(view_name):
    st.session_state.view = view_name



def append_message(role: str, content: str):
    """Append to display history and LangChain message list; trim to MAX_HISTORY."""
    st.session_state.messages.append({"role": role, "content": content})
    msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    st.session_state.lc_history.append(msg)
    if len(st.session_state.lc_history) > MAX_HISTORY:
        st.session_state.lc_history = st.session_state.lc_history[-MAX_HISTORY:]


# ────────────────────────────── Sidebar ───────────────────────────────────

def render_sidebar(pipeline, current_page_title, pages_dict):
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div class="sidebar-logo">
            <h2>CitizenLegal</h2>
            <p>RTI Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        for view_title in ["Home", "Resources", "Settings"]:
            is_active = "active" if current_page_title == view_title else ""
            if st.button(f"{view_title}", key=f"nav_{view_title}", use_container_width=True, type="secondary" if not is_active else "primary"):
                st.switch_page(pages_dict[view_title])

        st.spacer = st.empty()
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        if st.button("Start New Inquiry", use_container_width=True, type="primary"):
            st.session_state.messages   = []
            st.session_state.lc_history = []
            st.switch_page(pages_dict["Home"])

        st.divider()
        
        # Language Toggle
        st.session_state.hindi_mode = st.toggle(
            "अ Respond in Hindi",
            value=st.session_state.hindi_mode,
        )
        
        st.divider()
        st.markdown(f"#### {get_icon_svg('info', 18)} System Info", unsafe_allow_html=True)

        if pipeline:
            badges = [
                ("cpu", "LLM", GROQ_MODEL),
                ("database", "Chunks", str(pipeline["chunk_count"])),
            ]
            for icon_name, label, value in badges:
                st.markdown(
                    f'<div class="sidebar-badge"><span>{get_icon_svg(icon_name, 14)} {label}</span> <strong>{value}</strong></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown(
            f"<div style='text-align: center; opacity: 0.4; font-size: 0.7rem;'>"
            "SMAI Assignment 3 — T10.3<br>"
            "Groq · LangChain · ChromaDB</div>",
            unsafe_allow_html=True,
        )




# ─────────────────────────────── Main ─────────────────────────────────────

def main():
    init_session()

    # Load pipeline components (cached)
    pipeline = load_pipeline_components()

    if pipeline is None:
        st.error(
            "**Vector database not found.**\n\n"
            "Please run the ingestion pipeline first to build the knowledge base:\n"
            "```bash\npython ingest.py\n```",
            icon="📂"
        )
        st.stop()

    # Define Pages for native routing (prevents ghosting transitions)
    p_home = st.Page(lambda: render_home_view(pipeline), title="Home", url_path="home", default=True)
    p_resources = st.Page(render_resources_view, title="Resources", url_path="resources")
    p_settings = st.Page(render_settings_view, title="Settings", url_path="settings")

    pages_dict = {"Home": p_home, "Resources": p_resources, "Settings": p_settings}

    # Hide default sidebar nav, we will build our own
    current_page = st.navigation(list(pages_dict.values()), position="hidden")

    # Render custom sidebar
    render_sidebar(pipeline, current_page.title, pages_dict)

    # Run the selected page
    current_page.run()


def render_home_view(pipeline):
    # Welcome message / Quick Access when chat is empty
    if not st.session_state.messages:
        # Header / Hero
        st.markdown(f"""
        <div class="hero-card">
            <h1>{get_icon_svg('scale', 48, 'white')} RTI Assistant</h1>
            <p>Your intelligent companion for navigating the Right to Information Act. 
            Grounded in official government documentation to ensure accuracy and reliability.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### How can I help you today?")
        
        cards = [
            ("clipboard", "Application Process", "Step-by-step guidance on how to file an RTI request."),
            ("credit-card", "Fees & Payments", "Information on rates, exemptions, and payment methods."),
            ("clock", "Timelines", "Expected response periods and statutory deadlines."),
            ("gavel", "Legal Framework", "Understand your rights and specific exemptions."),
            ("zap", "Appeals", "Process for filing first and second appeals when requests are denied."),
            ("help", "Support & Help", "Get assistance with navigating this portal."),
        ]
        
        st.markdown('<div class="quick-access-grid">', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(cards):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="access-card">
                    <div class="icon-box">{get_icon_svg(icon, 24)}</div>
                    <div class="card-content">
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)



    # Render chat history
    for msg in st.session_state.messages:
        avatar = get_icon_data_uri("user", 24, "#2563eb") if msg["role"] == "user" else get_icon_data_uri("bot", 24, "#0f172a")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    placeholder = (
        "आरटीआई के बारे में कुछ भी पूछें…"
        if st.session_state.hindi_mode
        else "Ask anything about RTI..."
    )
    user_input = st.chat_input(placeholder)
    if not user_input:
        return

    # Sanitize
    user_input = user_input.strip()[:1000]

    with st.chat_message("user", avatar=get_icon_data_uri("user", 24, "#2563eb")):
        st.markdown(user_input)

    append_message("user", user_input)

    # Generate answer
    with st.chat_message("assistant", avatar=get_icon_data_uri("bot", 24, "#0f172a")):
        status = st.empty()

        status.markdown(f"{get_icon_svg('search', 16, '#2563eb')} *Consulting knowledge base...*", unsafe_allow_html=True)

        try:
            rag_chain = build_rag_chain(
                pipeline["llm"],
                pipeline["retriever"],
                st.session_state.hindi_mode,
            )
            history = st.session_state.lc_history[:-1]

            status.markdown(f"{get_icon_svg('zap', 16, '#2563eb')} *Synthesizing response...*", unsafe_allow_html=True)
            result = invoke_with_retry(rag_chain, {
                "input":        user_input,
                "chat_history": history,
            })

            status.empty()
            answer       = result.get("answer", "")
            source_docs  = result.get("source_documents", [])

            st.markdown(answer)

            if source_docs:
                unique_sources: dict = {}
                for doc in source_docs:
                    full_path = doc.metadata.get("source", "Unknown")
                    src  = os.path.basename(full_path)
                    page = doc.metadata.get("page", "?")
                    dt   = doc.metadata.get("doc_type", "")
                    key  = f"{src}  [{dt}]" if dt else src
                    # Use a tuple as key to keep track of filename vs display key
                    unique_sources.setdefault((src, key), set()).add(
                        str(int(page) + 1) if str(page).isdigit() else str(page)
                    )

                with st.expander(f"Reference Sources ({len(unique_sources)} documents)", expanded=False):
                    for (file_name, display_name), pages in unique_sources.items():
                        page_list = ", ".join(sorted(pages, key=lambda x: int(x) if x.isdigit() else 0))
                        
                        # Generate download link if file exists and get path
                        file_path = os.path.join("data", file_name)
                        download_html = ""
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, "rb") as f:
                                    b64_pdf = base64.b64encode(f.read()).decode('utf-8').replace('\\n', '')
                                download_html = (
                                    f'<a href="data:application/pdf;base64,{b64_pdf}" '
                                    f'download="{file_name}" style="margin-left: 10px; display: inline-flex; align-items: center; justify-content: center; text-decoration: none; color: #2563eb;" title="Download PDF">'
                                    f'{get_icon_svg("download", 18, "currentColor")}</a>'
                                )
                            except Exception:
                                pass
                        
                        st.markdown(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="display: flex; align-items: center; margin-right: 8px;">{get_icon_svg('file-text', 16)}</div>
                                <span style="font-weight: 500; font-size: 0.95rem; margin-right: 8px;">{display_name}</span>
                                <span style="color: #64748b; font-size: 0.9rem;">— page(s) {page_list}</span>
                                {download_html}
                            </div>
                        """, unsafe_allow_html=True)




            append_message("assistant", answer)

        except Exception as e:
            status.empty()
            st.error(f"❌ Error: {str(e)}")

def render_resources_view():
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; justify-content: center;">
                {get_icon_svg('book', 36)}
            </div>
            <h1 style="margin: 0; padding: 0;">Knowledge Resources</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("Access official RTI documentation and guides used to ground this assistant.")

    
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
        if not files:
            st.info("No PDF resources found in the data directory.")
        else:
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            for file_name in files:
                file_path = os.path.join(data_dir, file_name)
                
                download_link = ""
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            b64_pdf = base64.b64encode(f.read()).decode('utf-8').replace('\\n', '')
                        download_link = (
                            f'<a href="data:application/pdf;base64,{b64_pdf}" '
                            f'download="{file_name}" style="display: inline-flex; align-items: center; justify-content: center; text-decoration: none; color: #2563eb; padding: 8px; border-radius: 8px; transition: background 0.2s;" title="Download PDF" onmouseover="this.style.background=\'#eff6ff\'" onmouseout="this.style.background=\'transparent\'">'
                            f'{get_icon_svg("download", 22, "currentColor")}</a>'
                        )
                    except Exception:
                        pass

                # Card Container
                with st.container(border=True):
                    col1, col2 = st.columns([0.9, 0.1], vertical_alignment="center")
                    with col1:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div class="icon-box">{get_icon_svg('file-text', 20)}</div>
                            <div>
                                <div style="font-weight: 600; color: var(--text-main);">{file_name}</div>
                                <div style="font-size: 0.8rem; color: var(--text-muted);">Official PDF Document</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div style="display: flex; justify-content: flex-end;">{download_link}</div>', unsafe_allow_html=True)


    else:
        st.warning("Data directory not found.")


def render_settings_view():
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; justify-content: center;">
                {get_icon_svg('settings', 36)}
            </div>
            <h1 style="margin: 0; padding: 0;">Settings</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("Configure your experience with CitizenLegal RTI.")
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="display: flex; align-items: center; justify-content: center;">
                    {get_icon_svg('message', 22)}
                </div>
                <h3 style="margin: 0; padding: 0;">Language Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)


        st.session_state.hindi_mode = st.toggle(
            "Respond in Hindi",
            value=st.session_state.hindi_mode,
            help="When enabled, the assistant will respond in Devanagari script."
        )
        if st.session_state.hindi_mode:
            st.caption("Assistant will respond in Hindi (Devanagari).")
        else:
            st.caption("Assistant will respond in English.")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="display: flex; align-items: center; justify-content: center;">
                    {get_icon_svg('cpu', 22)}
                </div>
                <h3 style="margin: 0; padding: 0;">Model Configuration</h3>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)


        st.markdown(f"**Current Model:** `{GROQ_MODEL}`")
        st.markdown("**Temperature:** `0.0` (Fixed for consistency)")
        st.info("The model is grounded in official RTI documents to minimize hallucinations.")


if __name__ == "__main__":
    main()

