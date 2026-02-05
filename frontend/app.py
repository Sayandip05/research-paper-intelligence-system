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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-style layout
st.markdown("""
<style>
    /* Dark theme & ChatGPT feel */
    .stApp {
        background-color: #000000;
        color: #ECECF1;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }
    
    /* Center the main content when no messages */
    .main-content {
        max-width: 800px;
        margin: 0 auto;
        padding-top: 2rem;
    }
    
    /* Search options container */
    .search-options {
        background-color: #111111;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #333;
    }
    
    /* Result styling */
    .stMarkdown {
        font-family: 'Söhne', 'ui-sans-serif', 'system-ui', -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Noto Sans', sans-serif, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'; 
    }
    
    /* Chat input background */
    .stChatInputContainer {
        bottom: 20px;
        background-color: #000000;
    }
    
    /* Style for the info text above chat input */
    .chat-info-text {
        text-align: center;
        color: #9aa0a6;
        font-size: 0.85em;
        padding: 0.5rem;
        background-color: #111;
        border: 1px solid #333;
        border-radius: 10px 10px 0 0;
        border-bottom: none;
        margin-bottom: -1px;
    }
    
    /* Image cards */
    .image-card {
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #333;
        background: #111;
        margin-bottom: 0.5rem;
    }
    
    /* Sources expander */
    .streamlit-expanderHeader {
        background-color: #111 !important;
        color: #FFF !important;
        border: 1px solid #333;
    }
    
    /* Remove top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 10rem !important; /* Space for chat input */
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============== SIDEBAR ==============
with st.sidebar:
    st.title("📚 Research Papers")
    st.caption("Multimodal RAG System")
    st.divider()
    
    # ========== PDF UPLOAD ==========
    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Add a research paper",
        type=["pdf"],
        help="Upload PDF to index and query"
    )
    
    if uploaded_file:
        if st.button("📥 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Uploading..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    resp = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=30)
                    
                    if resp.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded!")
                        st.info("Processing in background...")
                        
                        # Poll for status
                        with st.spinner("Processing..."):
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
                                        # Clear upload to reset
                                        time.sleep(1)
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
    st.subheader("📊 Statistics")
    try:
        stats_resp = requests.get(f"{API_BASE_URL}/api/corpus/stats", timeout=5)
        img_stats = requests.get(f"{API_BASE_URL}/api/image-stats", timeout=5)
        
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", stats.get("total_documents", 0))
            with col2:
                if img_stats.status_code == 200:
                    st.metric("Images", img_stats.json().get("total_images", 0))
            st.caption(f"Total Chunks: {stats.get('total_chunks', 0)}")
    except:
        st.warning("Cannot fetch stats")
    
    st.divider()
    
    # ========== SYSTEM HEALTH ==========
    st.subheader("🔧 System Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("● Backend Online")
        else:
            st.error("● Backend Error")
    except:
        st.error("● Backend Offline")

# ============== MAIN CONTENT ==============

# Centered Header & Settings (Only show if no messages or reduced version)
if not st.session_state.messages:
    col_spacer_l, col_main, col_spacer_r = st.columns([1, 2, 1])
    with col_main:
        st.write("")
        st.write("")
        st.markdown("<h1 style='text-align: center;'>Research Paper Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #aaa;'>Ask in-depth questions about research papers using advanced Dense (BGE) + Sparse (BM42) multimodal retrieval with HITL validation</p>", unsafe_allow_html=True)
        st.write("")
        
        # Settings Container
        with st.container():


            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Search Mode**")
                search_mode = st.selectbox(
                    "Search Mode",
                    ["hybrid", "dense", "sparse"],
                    format_func=lambda x: {
                        "hybrid": "🔀 Hybrid (Best)",
                        "dense": "🧠 Dense Only",
                        "sparse": "📝 Keyword Only"
                    }[x],
                    label_visibility="collapsed",
                    key="search_mode_select"
                )
            with c2:
                st.markdown("**Depth (Top K)**")
                top_k = st.slider("Depth", 3, 10, 5, label_visibility="collapsed", key="top_k_slider")



else:
    # If messages exist, show a compact settings bar at the top
    with st.expander("⚙️ Search Settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            search_mode = st.selectbox(
                "Search Mode",
                ["hybrid", "dense", "sparse"],
                format_func=lambda x: {"hybrid": "🔀 Hybrid (Best)", "dense": "🧠 Dense Only", "sparse": "📝 Keyword Only"}[x],
                label_visibility="collapsed",
                key="search_mode_select_compact",
                index=["hybrid", "dense", "sparse"].index(st.session_state.get("last_mode", "hybrid")) 
            )
        with c2:
            top_k = st.slider("Depth", 3, 10, 5, label_visibility="collapsed", key="top_k_slider_compact", value=st.session_state.get("last_k", 5))

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Display Answer
            st.markdown(msg["content"])
            
            # Display Images if present
            if "images" in msg and msg["images"]:
                st.write("")
                st.markdown(f"**🖼️ Related Images ({len(msg['images'])})**")
                cols = st.columns(min(len(msg["images"]), 3))
                for i, img in enumerate(msg["images"]):
                    with cols[i % 3]:
                        image_id = img.get('image_id', '')
                        try:
                            # Direct fetch for display
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
                        except:
                            st.caption(f"Image {i+1} (Pg {img.get('page_number')})")
            
            # Display Sources if present
            if "sources" in msg and msg["sources"]:
                st.write("")
                with st.expander(f"📚 View {len(msg['sources'])} Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {source.get('paper_title', 'Unknown')}** (Section: {source.get('section_title', 'N/A')})")
                        st.caption(source.get("text", "")[:500] + "...")
                        st.divider()


# Chat Input (Fixed at bottom)
if prompt := st.chat_input("Ask a question about your papers..."):
    # Store settings used for this query
    st.session_state["last_mode"] = search_mode
    st.session_state["last_k"] = top_k

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({search_mode})..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/query",
                    json={
                        "question": prompt,
                        "similarity_top_k": top_k,
                        "response_mode": "compact",
                        "search_mode": search_mode
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer_text = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])
                    images = result.get("images", [])
                    
                    # Display Answer
                    st.markdown(answer_text)
                    
                    # Display Images
                    if images:
                        st.write("")
                        st.markdown(f"**🖼️ Related Images ({len(images)})**")
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
                                except:
                                    pass

                    # Display Sources
                    if sources:
                        st.write("")
                        with st.expander(f"📚 View {len(sources)} Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source.get('paper_title', 'Unknown')}** (Section: {source.get('section_title', 'N/A')})")
                                st.caption(source.get("text", "")[:500] + "...")
                                st.divider()
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "images": images
                    })
                    
                else:
                    error_msg = f"Error: {response.status_code}"
                    st.error(error_msg)
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
