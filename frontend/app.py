"""
Streamlit Frontend for Research Paper RAG - Hybrid Search

Features:
- PDF Upload with automatic hybrid processing
- Hybrid search (Dense + BM42 + RRF Fusion)
- Query with LLM synthesis

Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import time

# Config
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="centered"
)

# Header
st.title("📚 Research Paper Q&A")
st.caption("Hybrid RAG System - BM42 + Dense Search with RRF Fusion")

# Sidebar - PDF Upload (Auto-processing)
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
    help="PDF will be automatically processed with hybrid embeddings"
)

if uploaded_file:
    with st.sidebar.status("Uploading and processing...") as status:
        try:
            # Upload via API (auto-processing)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                status.update(label="✅ Processing started!", state="running")
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
                
                # Poll for status
                for i in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    status_resp = requests.get(
                        f"{API_BASE_URL}/api/upload/status/{uploaded_file.name}",
                        timeout=5
                    )
                    if status_resp.status_code == 200:
                        proc_status = status_resp.json()
                        if proc_status.get("status") == "completed":
                            status.update(label="✅ Processing complete!", state="complete")
                            st.sidebar.success(f"Created {proc_status.get('chunks_created', 0)} chunks")
                            break
                        elif proc_status.get("status") == "failed":
                            status.update(label="❌ Processing failed", state="error")
                            st.sidebar.error(proc_status.get("error", "Unknown error"))
                            break
            else:
                st.sidebar.error(f"Upload failed: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ Cannot connect to backend")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Sidebar - Corpus Info
st.sidebar.divider()
st.sidebar.header("📊 Corpus")
try:
    stats_resp = requests.get(f"{API_BASE_URL}/api/corpus/stats", timeout=5)
    if stats_resp.status_code == 200:
        stats = stats_resp.json()
        st.sidebar.metric("Total Chunks", stats.get("total_chunks", 0))
        st.sidebar.caption(f"Collection: {stats.get('collection', 'N/A')}")
        if stats.get("hybrid_enabled"):
            st.sidebar.success("🔀 Hybrid Search: Active")
        else:
            st.sidebar.info("📊 Dense Search Only")
except:
    pass

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

# Search mode toggle
search_mode = st.radio(
    "Search Mode",
    ["🔀 Hybrid Search (Dense + BM42)", "📊 Dense Only"],
    horizontal=True
)

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
    sections = st.multiselect(
        "Filter sections (optional)",
        ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"],
        default=[]
    )

# Submit button
if st.button("🔍 Get Answer", type="primary", disabled=not question):
    with st.spinner("Searching with hybrid retrieval..."):
        try:
            if "Hybrid" in search_mode:
                # Use hybrid search API
                search_resp = requests.post(
                    f"{API_BASE_URL}/api/search/hybrid",
                    json={
                        "query": question,
                        "top_k": top_k,
                        "sections": sections if sections else None
                    },
                    timeout=30
                )
                
                if search_resp.status_code == 200:
                    search_result = search_resp.json()
                    
                    st.info(f"🔀 Mode: {search_result.get('mode', 'hybrid')} | Papers: {search_result.get('paper_coverage', 0)}")
                    
                    # Now call LLM query for synthesis
                    response = requests.post(
                        f"{API_BASE_URL}/api/query",
                        json={
                            "question": question,
                            "similarity_top_k": top_k,
                            "response_mode": "compact"
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Check for HITL response
                        if result.get("status") == "human_review_required":
                            st.warning("⚠️ Human Review Required")
                            st.info(f"**Reason:** {result.get('reason', 'Low confidence')}")
                        else:
                            st.success("✅ Answer")
                            st.markdown(result.get("answer", "No answer generated"))
                            
                            # Sources from search
                            st.divider()
                            st.subheader(f"📖 Sources ({len(search_result.get('results', []))})")
                            for i, source in enumerate(search_result.get("results", []), 1):
                                with st.expander(f"Source {i}: {source.get('paper_title', 'Unknown')}"):
                                    st.caption(f"Section: {source.get('section', 'N/A')} | Score: {source.get('score', 0):.4f}")
                                    st.text(source.get("text", ""))
                else:
                    st.error(f"Search failed: {search_resp.status_code}")
            else:
                # Dense-only mode (original API)
                response = requests.post(
                    f"{API_BASE_URL}/api/query",
                    json={
                        "question": question,
                        "similarity_top_k": top_k,
                        "response_mode": "compact"
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Answer")
                    st.markdown(result.get("answer", "No answer generated"))
                    
                    sources = result.get("sources", [])
                    if sources:
                        st.divider()
                        st.subheader(f"📖 Sources ({len(sources)})")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source.get('paper_title', 'Unknown')}"):
                                st.caption(f"Section: {source.get('section_title', 'N/A')}")
                                st.text(source.get("text", "")[:500] + "...")
                else:
                    st.error(f"Query failed: {response.status_code}")
                    
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
st.caption("🔬 Hybrid RAG System v4.0 | BM42 + Dense + RRF Fusion")
