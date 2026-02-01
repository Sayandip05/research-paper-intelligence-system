"""
Multimodal Research Paper RAG - Streamlit Frontend

Features:
- PDF Upload with automatic multimodal processing
- Hybrid text search (Dense + BM42 + RRF Fusion)
- CLIP Image search (search images by text query)
- Query with LLM synthesis
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
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .result-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 Research Paper Intelligence System")
st.caption("Multimodal RAG - Hybrid Text Search + CLIP Image Search")

# ============== SIDEBAR ==============
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload a research paper",
        type=["pdf"],
        help="PDF will be auto-processed with text + image indexing"
    )
    
    if uploaded_file:
        with st.status("Uploading and processing...") as status:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=30)
                
                if response.status_code == 200:
                    status.update(label="✅ Processing started!", state="running")
                    st.success(f"Uploaded: {uploaded_file.name}")
                    
                    # Poll for status
                    for i in range(60):
                        time.sleep(2)
                        status_resp = requests.get(
                            f"{API_BASE_URL}/api/upload/status/{uploaded_file.name}",
                            timeout=5
                        )
                        if status_resp.status_code == 200:
                            proc_status = status_resp.json()
                            if proc_status.get("status") == "completed":
                                status.update(label="✅ Processing complete!", state="complete")
                                st.success(f"Created {proc_status.get('chunks_created', 0)} chunks")
                                break
                            elif proc_status.get("status") == "failed":
                                status.update(label="❌ Processing failed", state="error")
                                st.error(proc_status.get("error", "Unknown error"))
                                break
                else:
                    st.error(f"Upload failed: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Corpus Stats
    st.header("📊 Corpus Stats")
    try:
        stats_resp = requests.get(f"{API_BASE_URL}/api/corpus/stats", timeout=5)
        img_stats = requests.get(f"{API_BASE_URL}/api/image-stats", timeout=5)
        
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Text Chunks", stats.get("total_chunks", 0))
            with col2:
                if img_stats.status_code == 200:
                    st.metric("Images", img_stats.json().get("total_images", 0))
            
            if stats.get("hybrid_enabled"):
                st.success("🔀 Hybrid Search: Active")
            st.caption(f"Collection: {stats.get('collection', 'N/A')}")
    except:
        st.warning("Cannot fetch stats")
    
    st.divider()
    
    # System Health
    st.header("🔧 System Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("✅ Backend: Connected")
        else:
            st.error("❌ Backend: Error")
    except:
        st.error("❌ Backend: Offline")
        st.caption("Start: `cd backend && uvicorn app.main:app --reload`")

# ============== MAIN CONTENT ==============
tab1, tab2 = st.tabs(["💬 Text Q&A", "🖼️ Image Search"])

# ============== TAB 1: TEXT Q&A ==============
with tab1:
    st.header("Ask a Question")
    
    # Search mode
    search_mode = st.radio(
        "Search Mode",
        ["🔀 Hybrid (Dense + BM42)", "📊 Dense Only"],
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
    
    if st.button("🔍 Get Answer", type="primary", disabled=not question):
        with st.spinner("Searching and synthesizing..."):
            try:
                if "Hybrid" in search_mode:
                    # Hybrid search first
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
                
                # LLM Query
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
                    
                    if result.get("status") == "human_review_required":
                        st.warning("⚠️ Human Review Required")
                        st.info(f"**Reason:** {result.get('reason', 'Low confidence')}")
                    else:
                        st.success("✅ Answer")
                        st.markdown(result.get("answer", "No answer generated"))
                        
                        # Sources
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
                st.error("❌ Cannot connect to backend")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Example queries
    st.divider()
    st.subheader("💡 Example Questions")
    examples = [
        "What is the main contribution of LoRA?",
        "How does QLoRA reduce memory usage?",
        "What are the limitations of these methods?",
        "Compare LoRA and full fine-tuning"
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state["question"] = ex
                st.rerun()

# ============== TAB 2: IMAGE SEARCH ==============
with tab2:
    st.header("🖼️ Search Images by Text")
    st.caption("Uses CLIP (ViT-B/32) to find images matching your text query")
    
    # Image query input
    image_query = st.text_input(
        "Describe the image you're looking for:",
        placeholder="e.g., LoRA architecture diagram, training loss curve, model comparison chart"
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        img_top_k = st.slider("Max images", 1, 10, 3, key="img_k")
    with col2:
        min_score = st.slider("Min similarity", 0.1, 0.9, 0.15, key="min_score")  # Default 0.15 for better results
    
    if st.button("🔍 Search Images", type="primary", disabled=not image_query):
        with st.spinner("Searching with CLIP..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/image-search",
                    json={
                        "query": image_query,
                        "top_k": img_top_k,
                        "min_score": min_score
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    total = result.get("total_found", 0)
                    
                    if total > 0:
                        st.success(f"Found {total} matching images")
                        
                        for i, img_result in enumerate(result.get("results", []), 1):
                            with st.container():
                                st.markdown(f"""
                                **Image {i}** | Score: {img_result.get('score', 0):.3f}
                                - Paper: {img_result.get('paper_title', 'Unknown')}
                                - Page: {img_result.get('page_number', 'N/A')}
                                - Type: {img_result.get('metadata', {}).get('image_type', 'figure')}
                                """)
                                if img_result.get('caption'):
                                    st.caption(f"Caption: {img_result.get('caption')}")
                                st.divider()
                    else:
                        st.warning("No images found matching your query. Try different keywords.")
                else:
                    st.error(f"Search failed: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Example image queries
    st.divider()
    st.subheader("💡 Example Image Queries")
    img_examples = [
        "architecture diagram",
        "training loss curve",
        "comparison chart",
        "attention weights visualization"
    ]
    cols = st.columns(2)
    for i, ex in enumerate(img_examples):
        with cols[i % 2]:
            if st.button(ex, key=f"img_ex_{i}"):
                st.session_state["image_query"] = ex
                st.rerun()

# Footer
st.divider()
st.caption("🔬 Multimodal RAG v5.0 | BM42 + Dense + CLIP | Built with LlamaIndex & Qdrant")
