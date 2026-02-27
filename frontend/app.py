"""
AinSeba (আইনসেবা) - Streamlit Frontend
Clean, professional chat interface for the Bangladesh Legal Aid Assistant.

Connects to the FastAPI backend (Phase 5) via HTTP.

Run with:
    streamlit run frontend/app.py
"""

import uuid
import requests
import streamlit as st

# ============================================
# Configuration
# ============================================

API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="AinSeba - আইনসেবা",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: #1a5276;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }
    .app-header p {
        color: #666;
        font-size: 0.95rem;
    }

    /* Disclaimer banner */
    .disclaimer-banner {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #856404;
    }

    /* Source citation cards */
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #1a5276;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    .source-card .citation {
        font-weight: 600;
        color: #1a5276;
    }
    .source-card .score {
        color: #888;
        font-size: 0.78rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #f0f4f8;
    }
    .sidebar-section {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e0e0e0;
    }
    .sidebar-section h4 {
        color: #1a5276;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
    }

    /* Example question buttons */
    .stButton > button {
        text-align: left !important;
        font-size: 0.85rem !important;
    }

    /* Language badge */
    .lang-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .lang-en { background: #d4edda; color: #155724; }
    .lang-bn { background: #cce5ff; color: #004085; }
    .lang-banglish { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


# ============================================
# Session State Initialization
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st_{uuid.uuid4().hex[:12]}"
if "response_language" not in st.session_state:
    st.session_state.response_language = "auto"


# ============================================
# Helper Functions
# ============================================

def call_api(question: str, language: str = None, act_id: str = None, category: str = None) -> dict:
    """Call the FastAPI backend."""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "language": language,
            "use_reranker": True,
        }
        if act_id:
            payload["act_id"] = act_id
        if category:
            payload["category"] = category

        resp = requests.post(
            f"{API_BASE_URL}/api/query",
            json=payload,
            timeout=60,
        )

        if resp.status_code == 429:
            return {"error": "Rate limit exceeded. Please wait a moment and try again."}
        elif resp.status_code != 200:
            return {"error": f"Server error ({resp.status_code}): {resp.text}"}

        return resp.json()

    except requests.ConnectionError:
        return {"error": "Cannot connect to the AinSeba API. Make sure the server is running:\n\n`uvicorn src.api.app:app --reload --port 8000`"}
    except requests.Timeout:
        return {"error": "Request timed out. The server may be busy."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def check_api_health() -> dict:
    """Check if the backend is running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_available_sources() -> dict:
    """Fetch available law sources."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/sources", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def submit_feedback(query: str, answer: str, rating: int, comment: str = None):
    """Submit user feedback."""
    try:
        requests.post(
            f"{API_BASE_URL}/api/feedback",
            json={
                "query": query,
                "answer": answer,
                "rating": rating,
                "comment": comment,
                "session_id": st.session_state.session_id,
            },
            timeout=5,
        )
    except Exception:
        pass


def render_sources(sources: list):
    """Render source citations as expandable cards."""
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            citation = src.get("citation", "Unknown")
            sim = src.get("similarity_score", 0)
            rerank = src.get("rerank_score", 0)

            score_text = f"similarity: {sim:.3f}"
            if rerank:
                score_text += f" | rerank: {rerank:.3f}"

            st.markdown(
                f'<div class="source-card">'
                f'<span class="citation">[{i}] {citation}</span><br>'
                f'<span class="score">{score_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_language_badge(lang: str) -> str:
    """Return HTML for a language badge."""
    labels = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}
    css_class = f"lang-{lang}" if lang in labels else "lang-en"
    label = labels.get(lang, lang)
    return f'<span class="lang-badge {css_class}">{label}</span>'


# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("### Settings")

    # Language toggle
    lang_options = {"Auto-detect": "auto", "English": "en", "Bangla (বাংলা)": "bn"}
    selected_lang = st.selectbox(
        "Response Language",
        options=list(lang_options.keys()),
        index=0,
        help="Choose the language for responses. Auto-detect will match your query language.",
    )
    st.session_state.response_language = lang_options[selected_lang]

    # Filter by act
    sources_data = get_available_sources()
    act_filter = None
    category_filter = None

    if sources_data:
        acts = sources_data.get("acts", [])
        indexed_acts = [a for a in acts if a.get("indexed")]
        categories = sources_data.get("categories", [])

        if indexed_acts:
            act_names = ["All Acts"] + [a["name"] for a in indexed_acts]
            selected_act = st.selectbox("Filter by Act", act_names, index=0)
            if selected_act != "All Acts":
                act_filter = next(
                    (a["id"] for a in indexed_acts if a["name"] == selected_act), None
                )

        if categories:
            cat_names = ["All Categories"] + categories
            selected_cat = st.selectbox("Filter by Category", cat_names, index=0)
            if selected_cat != "All Categories":
                category_filter = selected_cat

    st.markdown("---")

    # Example questions
    st.markdown("### Example Questions")

    example_questions = [
        "What is the penalty for theft?",
        "What are the maximum working hours per day?",
        "My employer hasn't paid me for 3 months. What can I do?",
        "চুরির শাস্তি কী?",
        "amar malik betan dey nai, ki korbo?",
        "What rights does a tenant have?",
        "How do I file a consumer complaint?",
    ]

    for q in example_questions:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")

    # API status
    health = check_api_health()
    if health:
        doc_count = health.get("vector_store_documents", 0)
        st.success(f"API Connected | {doc_count} documents indexed")
    else:
        st.error("API Offline — Start with:\n`uvicorn src.api.app:app --reload`")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"st_{uuid.uuid4().hex[:12]}"
        st.rerun()


# ============================================
# Main Chat Area
# ============================================

# Header
st.markdown(
    '<div class="app-header">'
    '<h1>AinSeba আইনসেবা</h1>'
    '<p>Bangladesh Legal Aid Assistant — Ask questions about Bangladesh law in English, Bangla, or Banglish</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Disclaimer
st.markdown(
    '<div class="disclaimer-banner">'
    '<strong>Disclaimer:</strong> This tool provides legal information for educational purposes only. '
    'It does not constitute legal advice. For specific legal matters, please consult a qualified lawyer.'
    '</div>',
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

        # Show language info
        if msg["role"] == "assistant" and msg.get("detected_language"):
            lang = msg["detected_language"]
            translated = msg.get("was_translated", False)
            badge = render_language_badge(lang)
            extra = " (translated)" if translated else ""
            st.markdown(
                f'<div style="font-size: 0.78rem; color: #888; margin-top: 0.3rem;">'
                f'Detected: {badge}{extra}</div>',
                unsafe_allow_html=True,
            )

        # Feedback buttons
        if msg["role"] == "assistant" and msg.get("answer"):
            msg_key = f"fb_{hash(msg['answer'][:50])}"
            cols = st.columns([1, 1, 8])
            with cols[0]:
                if st.button("👍", key=f"{msg_key}_up", help="Helpful"):
                    submit_feedback(msg.get("query", ""), msg["answer"], 5)
                    st.toast("Thanks for the feedback!")
            with cols[1]:
                if st.button("👎", key=f"{msg_key}_down", help="Not helpful"):
                    submit_feedback(msg.get("query", ""), msg["answer"], 1)
                    st.toast("Thanks — we'll work on improving!")

# Handle pending question from sidebar examples
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching Bangladesh law..."):
            lang = st.session_state.response_language
            result = call_api(
                question,
                language=lang if lang != "auto" else None,
                act_id=act_filter,
                category=category_filter,
            )

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["error"],
            })
        else:
            answer = result.get("answer", "No answer received.")
            st.markdown(answer)

            sources = result.get("sources", [])
            render_sources(sources)

            detected = result.get("detected_language", "en")
            badge = render_language_badge(detected)
            translated = result.get("was_translated", False)
            extra = " (translated)" if translated else ""
            st.markdown(
                f'<div style="font-size: 0.78rem; color: #888; margin-top: 0.3rem;">'
                f'Detected: {badge}{extra}</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "answer": answer,
                "sources": sources,
                "query": question,
                "detected_language": detected,
                "was_translated": translated,
            })

    st.rerun()

# Chat input
if question := st.chat_input("Ask a legal question in English, Bangla, or Banglish..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching Bangladesh law..."):
            lang = st.session_state.response_language
            result = call_api(
                question,
                language=lang if lang != "auto" else None,
                act_id=act_filter,
                category=category_filter,
            )

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["error"],
            })
        else:
            answer = result.get("answer", "No answer received.")
            st.markdown(answer)

            sources = result.get("sources", [])
            render_sources(sources)

            detected = result.get("detected_language", "en")
            badge = render_language_badge(detected)
            translated = result.get("was_translated", False)
            extra = " (translated)" if translated else ""
            st.markdown(
                f'<div style="font-size: 0.78rem; color: #888; margin-top: 0.3rem;">'
                f'Detected: {badge}{extra}</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "answer": answer,
                "sources": sources,
                "query": question,
                "detected_language": detected,
                "was_translated": translated,
            })
