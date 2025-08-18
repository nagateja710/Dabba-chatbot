#paste it in test.py to test
import os
import uuid
import streamlit as st

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Dabba Chatbot")
st.write("Multi‑Session Chat with Optional PDF RAG powered by LangChain’s message history")

# ================== STATE INIT ==================
if "store" not in st.session_state:
    st.session_state.store = {}         # LC histories per session_id
if "disp" not in st.session_state:
    st.session_state.disp = {}          # UI transcripts per session_id
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "newchat"
if "newchat_counter" not in st.session_state:
    st.session_state.newchat_counter = 1
if "files_hash" not in st.session_state:
    st.session_state.files_hash = None  # to rebuild KB only when uploads change

from utils import *

bind_messages_display_to_current_session()

# ================== UPLOAD & INDEX ==================
uploaded_files = st.file_uploader(
    "Choose PDF file(s) (optional)",
    type="pdf",
    accept_multiple_files=True,
)


def compute_files_hash(files) -> str | None:
    if not files:
        return None
    names = tuple(sorted(f.name for f in files))
    return str(names)

# Only rebuild when the set of files changes
current_hash = compute_files_hash(uploaded_files)
rebuild = current_hash is not None and current_hash != st.session_state.files_hash
if current_hash is None:
    clear_vector_db()
if rebuild:

    documents = []
    for uf in uploaded_files:
        temp_name = f"./tmp_{uuid.uuid4().hex}.pdf"
        with open(temp_name, "wb") as f:
            f.write(uf.getvalue())
        docs = PyPDFLoader(temp_name).load()
        documents.extend(docs)
        os.remove(temp_name)


    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    st.session_state.vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    st.session_state.files_hash = current_hash
    st.success(f"Indexed {len(splits)} chunks from {len(uploaded_files)} file(s).")

elif current_hash is None:
    # No files: do not reset vectorstore/retriever; just leave them as-is
    pass

# ================== CHAINS ==================
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

from chains import *

if st.session_state.retriever is None:
    core_chain,output_key=chain(llm)
else:
    core_chain,output_key=rag_chain(llm)

conversational_chain = RunnableWithMessageHistory(
    core_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key=output_key,
)



# ================== SIDEBAR CONTROLS ==================
def get_options():
    keys = list(st.session_state.disp.keys())
    if st.session_state.session_id not in keys:
        keys.append(st.session_state.session_id)
    return sorted(set(keys))

with st.sidebar:
    st.write(f"Current session: {st.session_state.session_id}")

    # Session switcher
    options = get_options()
    selected = st.selectbox(
        "Select session",
         options=options,
         index=options.index(st.session_state.session_id) if st.session_state.session_id in options else 0,
          key="session_selector",
    )
    if selected != st.session_state.session_id:
        if not uploaded_files:
          switch_session(selected)
          st.success(f"Switched to: {selected}")
        else:
          st.error('First delete uploaded file!')

    if st.button("New session"):
        if not uploaded_files:
          create_new_session()
          st.rerun()
          st.success(f"Created: {st.session_state.session_id}")
        else:
          st.error('First delete uploaded file!')


    with st.expander("Debug: Chat history", expanded=True):
        hist = get_session_history(st.session_state.session_id)
        st.caption(f"History length: {len(hist.messages)/2}")
        show_preview = st.checkbox("Show last 2 conversations", value=False)
        if show_preview:
            recent = hist.messages[-4:]
            for i, m in enumerate(reversed(recent), 1):
                role = getattr(m, "type", getattr(m, "role", "message"))
                content = getattr(m, "content", str(m))
                snippet = content if len(content) <= 200 else content[:200] + "..."
                st.markdown(f"- {i}. [{role}] {snippet}")
    # st.write(f'{current_hash}')


# ================== CHAT UI ==================
st.divider()

with st.form("ask"):
    user_input = st.text_input("Your question:", key="user_input")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    # Store user turn (newest-first UI)
    st.session_state.messages_display.insert(0, (user_input, ""))

    # Invoke chain with consistent session_id
    try:
        raw = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )

        show_top_chunks_with_page(raw, title="Top retrieved chunks")
        answer = extract_answer(raw)
        st.session_state.messages_display[0] = (user_input, answer)

    except Exception as e:
        answer = f"[Error] {e}"


# Render transcript (newest first)
for user_msg, bot_msg in st.session_state.messages_display:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Assistant:** {bot_msg}")
    st.markdown("---")
