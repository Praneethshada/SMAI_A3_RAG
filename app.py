import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = "chroma_db"
DEFAULT_GROQ_MODEL = "llama3-8b-8192"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

st.set_page_config(page_title="RTI Assistant", page_icon="📝")

@st.cache_resource
def load_rag_pipeline():
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load Vector Store
    if not os.path.exists(CHROMA_PATH):
        st.error("Vector database not found. Please run ingest.py first.")
        st.stop()
        
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 3. Setup Retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # 4. Setup LLM
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    
    # 5. Setup Prompt and Chain
    system_prompt = (
        "You are an assistant for question-answering tasks specifically about Indian Right to Information (RTI) services.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer based on the context, just say that you don't know. Do not make up answers.\n"
        "Keep the answer concise and helpful.\n"
        "\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

st.title("🇮🇳 RTI Assistant Chatbot")
st.markdown("Ask any questions about filing an RTI (Right to Information), fees, processes, and rules.")
st.caption(f"Using Groq model: {GROQ_MODEL}")

rag_chain = load_rag_pipeline()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How do I file an RTI online?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                source_documents = response.get("context", [])
                
                # Format sources
                if source_documents:
                    sources_str = "\n\n**Sources:**\n"
                    unique_sources = set()
                    for doc in source_documents:
                        source = doc.metadata.get("source", "Unknown PDF")
                        page = doc.metadata.get("page", "Unknown Page")
                        source_name = os.path.basename(source)
                        unique_sources.add(f"- {source_name} (Page {page})")
                    
                    for src in unique_sources:
                        sources_str += src + "\n"
                        
                    answer += sources_str
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_text = str(e)
                if "404" in error_text or "not found" in error_text.lower():
                    friendly_error = (
                        f"The configured Groq model '{GROQ_MODEL}' is unavailable. "
                        "Set GROQ_MODEL in .env to a valid model, then restart Streamlit."
                    )
                    st.error(friendly_error)
                    st.session_state.messages.append({"role": "assistant", "content": friendly_error})
                elif "429" in error_text or "rate limit" in error_text.lower() or "rate_limit" in error_text.lower():
                    friendly_error = (
                        "Your Groq API quota is exhausted or rate-limited (429). "
                        "Try again later, or check your API key limits."
                    )
                    st.error(friendly_error)
                    st.session_state.messages.append({"role": "assistant", "content": friendly_error})
                else:
                    st.error(f"Error generating response: {error_text}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {error_text}"})
