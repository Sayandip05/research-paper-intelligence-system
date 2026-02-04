"""
Multimodal Research Paper RAG - Streamlit Frontend

🆕 Unified search: Returns text + images together
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

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
    .image-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        background: #f9f9f9;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 Research Paper Intelligence System")
st.caption("🆕 Unified Multimodal RAG: Hybrid Text + CLIP Images")

# ============== SIDEBAR ==============
with st.sidebar:
    # ========== PDF UPLOAD ==========
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Add a research paper",
        type=["pdf"],
        help="Upload PDF to index and query"
    )
    
    if uploaded_file:
        if st.button("📥 Process PDF", type="primary"):
            with st.spinner("Uploading..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    resp = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=30)
                    
                    if resp.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded!")
                        st.info("Processing in background...")
                        
                        # Poll for status
                        with st.spinner("Processing..."):
                            import time
                            for _ in range(60):  # Max 60 seconds
                                time.sleep(2)
                                status_resp = requests.get(
                                    f"{API_BASE_URL}/api/upload/status/{uploaded_file.name}",
                                    timeout=5
                                )
                                if status_resp.status_code == 200:
                                    status = status_resp.json()
                                    if status.get("status") == "completed":
                                        st.success(f"✅ Done! {status.get('chunks_created', 0)} chunks created")
                                        st.rerun()
                                        break
                                    elif status.get("status") == "failed":
                                        st.error(f"❌ Failed: {status.get('error', 'Unknown error')}")
                                        break
                    else:
                        st.error(f"Upload failed: {resp.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # ========== CORPUS STATS ==========
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
            
            st.success("🔀 Hybrid Search: Active")
            st.success("🖼️ Image Search: Active")
    except:
        st.warning("Cannot fetch stats")
    
    st.divider()
    
    # ========== SYSTEM HEALTH ==========
    st.header("🔧 System Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("✅ Backend: Connected")
        else:
            st.error("❌ Backend: Error")
    except:
        st.error("❌ Backend: Offline")

# ============== MAIN CONTENT ==============
st.header("💬 Research Paper Q&A")

# Initialize session state for results
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ========== RESULTS AREA (TOP) ==========
if st.session_state.last_result:
    result = st.session_state.last_result
    
    # Answer
    st.success("✅ Answer")
    st.markdown(result.get("answer", "No answer"))
    
    # Images
    images = result.get("images", [])
    if images:
        st.divider()
        st.subheader(f"🖼️ Related Images ({len(images)})")
        cols = st.columns(min(len(images), 3))
        for i, img in enumerate(images):
            with cols[i % 3]:
                image_id = img.get('image_id', '')
                try:
                    img_response = requests.get(
                        f"{API_BASE_URL}/api/image-by-id/{image_id}",
                        timeout=10
                    )
                    if img_response.status_code == 200:
                        st.image(
                            img_response.content,
                            caption=f"Page {img.get('page_number', 'N/A')} | {img.get('image_type', 'figure')}",
                            use_container_width=True
                        )
                    else:
                        st.info(f"🖼️ Image {i+1}\n📄 {img.get('paper_title', '')[:25]}...")
                except:
                    st.info(f"🖼️ Image {i+1}\n📍 Page {img.get('page_number', 'N/A')}")
                st.caption(f"Score: {img.get('score', 0):.2f}")
    
    # Sources
    sources = result.get("sources", [])
    if sources:
        st.divider()
        st.subheader(f"📖 Text Sources ({len(sources)})")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source.get('paper_title', 'Unknown')}"):
                st.caption(f"Section: {source.get('section_title', 'N/A')}")
                st.text(source.get("text", "")[:500] + "...")

# ========== INPUT AREA (BOTTOM) ==========
st.divider()

# Options row
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Text sources", 3, 10, 5)
with col2:
    search_mode = st.selectbox(
        "Search Mode",
        ["hybrid", "dense", "sparse"],
        format_func=lambda x: {"hybrid": "🔀 Hybrid (Both)", "dense": "🧠 Dense (BGE)", "sparse": "📝 BM42 (Sparse)"}[x],
        index=0
    )

# Question input (at bottom like ChatGPT)
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Ask about your research papers:",
        placeholder="What is LoRA? How does the Transformer architecture work?",
        label_visibility="collapsed"
    )
with col_btn:
    submit = st.button("🔍", type="primary", disabled=not question)

if submit and question:
    with st.spinner(f"Searching with {search_mode} mode..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/query",
                json={
                    "question": question,
                    "similarity_top_k": top_k,
                    "response_mode": "compact",
                    "search_mode": search_mode
                },
                timeout=60
            )
            
            if response.status_code == 200:
                st.session_state.last_result = response.json()
                st.rerun()
            else:
                st.error(f"Query failed: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.caption("🔬 Multimodal RAG v5.1 | Hybrid Text (Dense+BM42) + CLIP Images | Built with LlamaIndex & Qdrant")
