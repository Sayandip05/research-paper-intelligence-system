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
st.header("💬 Ask a Question")
st.caption("Returns text answers + related images automatically")

# Question input
question = st.text_input(
    "Your question:",
    placeholder="e.g., What is the LoRA architecture? Show me diagrams."
)

# Options
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Text sources", 3, 10, 5)
with col2:
    sections = st.multiselect(
        "Filter sections (optional)",
        ["Abstract", "Introduction", "Methods", "Results", "Discussion"],
        default=[]
    )

if st.button("🔍 Get Answer", type="primary", disabled=not question):
    with st.spinner("Searching text + images..."):
        try:
            # Query API (returns text + images)
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
                    # ========== ANSWER ==========
                    st.success("✅ Answer")
                    st.markdown(result.get("answer", "No answer"))
                    
                    # ========== IMAGES (IF ANY) ==========
                    images = result.get("images", [])
                    if images:
                        st.divider()
                        st.subheader(f"🖼️ Related Images ({len(images)})")
                        
                        # Display images in grid
                        cols = st.columns(min(len(images), 3))
                        for i, img in enumerate(images):
                            with cols[i % 3]:
                                image_id = img.get('image_id', '')
                                
                                # Try to fetch and display actual image
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
                                        # Fallback to metadata display
                                        st.info(f"🖼️ Image {i+1}\n📄 {img.get('paper_title', '')[:25]}...\n📍 Page {img.get('page_number', 'N/A')}")
                                except:
                                    st.info(f"🖼️ Image {i+1}\n📄 {img.get('paper_title', '')[:25]}...\n📍 Page {img.get('page_number', 'N/A')}")
                                
                                st.caption(f"Score: {img.get('score', 0):.2f}")
                    
                    # ========== TEXT SOURCES ==========
                    sources = result.get("sources", [])
                    if sources:
                        st.divider()
                        st.subheader(f"📖 Text Sources ({len(sources)})")
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
    "What is LoRA? Show architecture.",
    "How does QLoRA reduce memory?",
    "Compare LoRA and full fine-tuning",
    "Show me training loss curves"
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}"):
            st.rerun()

# Footer
st.divider()
st.caption("🔬 Multimodal RAG v5.1 | Hybrid Text (Dense+BM42) + CLIP Images | Built with LlamaIndex & Qdrant")
