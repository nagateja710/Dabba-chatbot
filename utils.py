
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# ==============================
# Utility Functions
# ==============================
def _is_invalid_session_id(sid: str) -> bool:
    return (sid is None) or (not isinstance(sid, str)) or (sid.strip() == "")


def _generate_new_session_id() -> str:
    sid = f"newchat_{st.session_state.newchat_counter:03d}"
    st.session_state.newchat_counter += 1
    return sid


def bind_messages_display_to_current_session():
    """Ensure messages_display points to the transcript list of the active session."""
    sid = st.session_state.session_id
    if _is_invalid_session_id(sid):
        sid = _generate_new_session_id()
        st.session_state.session_id = sid
    if sid not in st.session_state.disp:
        st.session_state.disp[sid] = []
    st.session_state.messages_display = st.session_state.disp[sid]

def clear_vector_db():
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.files_hash = None



# # ================== HELPERS ==================
def extract_answer(raw):
    """Normalize any chain output to a displayable string."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return raw.get("answer") or raw.get("text") or raw.get("output") or ""
    content = getattr(raw, "content", None)
    if content is not None:
        return content
    to_dict = getattr(raw, "model_dump", None) or getattr(raw, "dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            return d.get("answer") or d.get("text") or d.get("output") or d.get("content") or ""
        except Exception:
            return ""
    return str(raw)



def switch_session(new_sid: str):
    """Switch active session and rebind transcript; ensure LC history exists."""
    st.session_state.user_input = ""
    if _is_invalid_session_id(new_sid):
        return
    st.session_state.session_id = new_sid
    if new_sid not in st.session_state.disp:
        st.session_state.disp[new_sid] = []
    st.session_state.messages_display = st.session_state.disp[new_sid]
    if new_sid not in st.session_state.store:
        st.session_state.store[new_sid] = ChatMessageHistory()

    clear_vector_db()
    return True

def create_new_session():
    new_sid = _generate_new_session_id()
    st.session_state.disp[new_sid] = []
    st.session_state.store[new_sid] = ChatMessageHistory()
    switch_session(new_sid)


def show_top_chunks_with_page(raw, title="Top retrieved sources", max_chunks=3):
    # raw is expected to be a dict with "context" = list[Document]
    if(st.session_state.retriever==None):
      return
    docs = raw.get("context", []) if isinstance(raw, dict) else []
    with st.sidebar.expander(title, expanded=True):
        if not docs:
            st.caption("No retrieved chunks.")
            return
        # Take exactly the first N chunks returned by the retriever/chain
        for i, doc in enumerate(docs[:max_chunks], 1):
            meta = getattr(doc, "metadata", {}) or {}
            page = meta.get("page") or meta.get("page_number")
            content = getattr(doc, "page_content", "") or ""
            # Header shows chunk index and source page number
            header = f"Chunk {i} — Page {page}" if page is not None else f"Chunk {i} — Page ?"
            st.markdown(f"**{header}**")
            st.code(content)