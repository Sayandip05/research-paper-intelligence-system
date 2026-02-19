"""
Multimodal Research Paper RAG - Streamlit Frontend

🆕 ChatGPT-style sessions with MongoDB persistence
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
    
    /* Result styling */
    .stMarkdown {
        font-family: 'Söhne', 'ui-sans-serif', 'system-ui', -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Noto Sans', sans-serif, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'; 
    }
    
    /* Chat input background */
    .stChatInputContainer {
        bottom: 20px;
        background-color: #000000;
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
        padding-bottom: 10rem !important;
    }
    
    /* Session buttons in sidebar */
    .session-btn {
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 2px;
        cursor: pointer;
        color: #ccc;
        font-size: 0.9em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions_list" not in st.session_state:
    st.session_state.sessions_list = []


# ── Helper Functions ──────────────────────────────────────────────
def fetch_sessions():
    """Fetch all sessions from backend"""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/sessions", timeout=5)
        if resp.status_code == 200:
            st.session_state.sessions_list = resp.json().get("sessions", [])
    except:
        st.session_state.sessions_list = []


def load_session(session_id):
    """Load a session's messages from backend"""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/sessions/{session_id}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.active_session_id = session_id
            st.session_state.messages = data.get("messages", [])
            return True
    except:
        pass
    return False


def create_new_session():
    """Create a new session via backend"""
    try:
        resp = requests.post(f"{API_BASE_URL}/api/sessions", json={}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.active_session_id = data["session_id"]
            st.session_state.messages = []
            return data["session_id"]
    except:
        pass
    return None


def delete_session(session_id):
    """Delete a session"""
    try:
        requests.delete(f"{API_BASE_URL}/api/sessions/{session_id}", timeout=5)
    except:
        pass


# ── Fetch sessions on load ────────────────────────────────────────
fetch_sessions()


# ============== SIDEBAR ==============
with st.sidebar:
    st.title("📚 Research Papers")
    st.caption("Multimodal RAG System")
    st.divider()
    
    # ========== NEW CHAT BUTTON ==========
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        create_new_session()
        st.rerun()
    
    st.divider()
    
    # ========== SESSION LIST ==========
    st.subheader("💬 Chat History")
    
    if st.session_state.sessions_list:
        for session in st.session_state.sessions_list:
            sid = session["session_id"]
            title = session.get("title", "New Chat")
            is_active = sid == st.session_state.active_session_id
            
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                label = f"{'▶ ' if is_active else ''}{title}"
                if st.button(label, key=f"session_{sid}", use_container_width=True):
                    load_session(sid)
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{sid}"):
                    delete_session(sid)
                    if st.session_state.active_session_id == sid:
                        st.session_state.active_session_id = None
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No sessions yet. Click 'New Chat' to start.")
    
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
                            for _ in range(60):
                                time.sleep(2)
                                status_resp = requests.get(
                                    f"{API_BASE_URL}/api/upload/status/{uploaded_file.name}",
                                    timeout=5
                                )
                                if status_resp.status_code == 200:
                                    status = status_resp.json()
                                    if status.get("status") == "completed":
                                        st.success(f"✅ Done! {status.get('chunks_created', 0)} chunks created")
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
    
    st.divider()
    
    # ========== VOICE QUERY ==========
    st.subheader("🎤 Voice Query")
    st.caption("Record audio to ask a question")
    audio_data = st.audio_input("Record your question", key="voice_recorder")
    
    if audio_data is not None:
        if st.button("🚀 Send Voice Query", type="primary", use_container_width=True, key="send_voice"):
            # Auto-create session if none
            if not st.session_state.active_session_id:
                create_new_session()
            
            with st.spinner("🎤 Transcribing & Querying..."):
                try:
                    files = {"audio": ("recording.wav", audio_data, "audio/wav")}
                    form_data = {
                        "search_mode": st.session_state.get("last_mode", "hybrid"),
                        "similarity_top_k": str(st.session_state.get("last_k", 5)),
                    }
                    resp = requests.post(
                        f"{API_BASE_URL}/api/query/voice",
                        files=files,
                        data=form_data,
                        timeout=60
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        transcribed = result.get("transcribed_text", "")
                        answer = result.get("answer", "No answer.")
                        sources = result.get("sources", [])
                        images = result.get("images", [])
                        
                        # Save to session messages
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"🎤 {transcribed}"
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "images": images
                        })
                        
                        # Also save to backend session
                        session_id = st.session_state.active_session_id
                        if session_id:
                            try:
                                requests.post(
                                    f"{API_BASE_URL}/api/sessions/{session_id}/query",
                                    json={
                                        "question": transcribed,
                                        "similarity_top_k": st.session_state.get("last_k", 5),
                                        "search_mode": st.session_state.get("last_mode", "hybrid")
                                    },
                                    timeout=60
                                )
                            except:
                                pass
                        
                        st.success(f"✅ Transcribed: {transcribed}")
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"❌ {detail}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============== MAIN CONTENT ==============

# Centered Header & Settings (Only show if no messages)
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
    role = msg.get("role", "user")
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg["content"])
        else:
            # Display Answer
            st.markdown(msg["content"])
            
            # Display Images if present
            images = msg.get("images") or []
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
                            st.caption(f"Image {i+1} (Pg {img.get('page_number')})")
            
            # Display Sources if present
            sources = msg.get("sources") or []
            if sources:
                st.write("")
                with st.expander(f"📚 View {len(sources)} Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}. {source.get('paper_title', 'Unknown')}** (Section: {source.get('section_title', 'N/A')})")
                        st.caption(source.get("text", "")[:500] + "...")
                        st.divider()


# Chat Input (Fixed at bottom)
if prompt := st.chat_input("Ask a question about your papers..."):
    # Store settings used for this query
    st.session_state["last_mode"] = search_mode
    st.session_state["last_k"] = top_k

    # Auto-create session if none active
    if not st.session_state.active_session_id:
        create_new_session()

    session_id = st.session_state.active_session_id

    # Add user message to local state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response via session endpoint
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({search_mode})..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/sessions/{session_id}/query",
                    json={
                        "question": prompt,
                        "similarity_top_k": top_k,
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
                    
                    # Save to local history
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
