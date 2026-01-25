"""
Minimal Streamlit Frontend for Research Paper RAG

This is a TEMPORARY UI for demo/testing.
Production UI will be custom HTML/CSS/JS later.

Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import os
from pathlib import Path

# Config
API_BASE_URL = "http://localhost:8000"
CORPUS_FOLDER = Path(__file__).parent.parent / "corpus"

# Page config
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="centered"
)

# Header
st.title("📚 Research Paper Q&A")
st.caption("Temporary demo UI - Week 3 RAG System")

# Sidebar - PDF Upload
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
    help="Upload a PDF to add to the corpus"
)

if uploaded_file:
    # Save uploaded file to corpus folder
    os.makedirs(CORPUS_FOLDER, exist_ok=True)
    save_path = CORPUS_FOLDER / uploaded_file.name
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
    st.sidebar.info("⚠️ Run `python build_corpus.py` to index this PDF")

# Sidebar - API Health
st.sidebar.divider()
st.sidebar.header("🔧 System Status")

try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=3)
    if health_response.status_code == 200:
        st.sidebar.success("✅ Backend: Connected")
    else:
        st.sidebar.error("❌ Backend: Error")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Backend: Offline")
    st.sidebar.caption("Start with: `cd backend && uvicorn app.main:app --reload`")

# Main - Query Section
st.header("💬 Ask a Question")

# Question input
question = st.text_input(
    "Your question about the research papers:",
    placeholder="e.g., What are the limitations of LoRA?"
)

# Options
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Number of sources", 1, 10, 5)
with col2:
    response_mode = st.selectbox("Response mode", ["compact", "refine", "tree_summarize"])

# Submit button
if st.button("🔍 Get Answer", type="primary", disabled=not question):
    with st.spinner("Thinking..."):
        try:
            # Call API
            response = requests.post(
                f"{API_BASE_URL}/api/query",
                json={
                    "question": question,
                    "similarity_top_k": top_k,
                    "response_mode": response_mode
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check for HITL response
                if result.get("status") == "human_review_required":
                    st.warning("⚠️ Human Review Required")
                    st.info(f"**Reason:** {result.get('reason', 'Low confidence')}")
                    st.caption(f"Intent: {result.get('intent', 'N/A')}")
                    st.caption(f"Suggestion: {result.get('suggestion', 'Please rephrase your question.')}")
                else:
                    # Normal answer
                    st.success("✅ Answer")
                    st.markdown(result.get("answer", "No answer generated"))
                    
                    # Sources
                    sources = result.get("sources", [])
                    if sources:
                        st.divider()
                        st.subheader(f"📖 Sources ({len(sources)})")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source.get('paper_title', 'Unknown')}"):
                                st.caption(f"Section: {source.get('section', 'N/A')}")
                                st.caption(f"Score: {source.get('score', 'N/A'):.3f}")
                                st.text(source.get("text", "")[:500] + "...")
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is FastAPI running?")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The LLM might be slow.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Example queries
st.divider()
st.subheader("💡 Example Questions")

example_queries = [
    "What is the main contribution of LoRA?",
    "How does QLoRA reduce memory usage?",
    "What are the limitations of these methods?",
    "Compare LoRA and full fine-tuning",
    "Give a brief summary of the paper"
]

cols = st.columns(2)
for i, example in enumerate(example_queries):
    with cols[i % 2]:
        if st.button(example, key=f"ex_{i}"):
            st.session_state["question"] = example
            st.rerun()

# Footer
st.divider()
st.caption("🔬 Week 3: Event-Driven Multi-Agent Workflow | Temporary Demo UI")
