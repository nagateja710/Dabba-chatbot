# DABBA CHATBOT

Multi‑Session Chat with Optional PDF RAG powered by LangChain’s message history

## Overview
DABBA Chatbot is a lightweight Streamlit app for conversational AI with optional Retrieval-Augmented Generation (RAG) over uploaded PDFs. It builds an on‑the‑fly vector index using MiniLM embeddings and Chroma, keeps isolated multi‑session chat histories via LangChain’s message history, and can display top retrieved chunks for transparency.

## Repository contents
- requirements.txt
- app.py (main)
- chains.py
- utils.py
- a.env

## Quick start

### 1) Add API keys
Create or open the a.env file in the repo root and paste the following (replace with a real key):

GROQ_API_KEY='XXXXXX'

If other environment variables are needed (e.g., model names or persistence paths), add them here as well.

### 2) Create a virtual environment (recommended)
- macOS/Linux:
  - python -m venv .venv
  - source .venv/bin/activate
- Windows (PowerShell):
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1

### 3) Install requirements
pip install -U pip
pip install -r requirements.txt


### 4) Run the app
- Ensure the environment variables are loaded (e.g., by your IDE, shell profile, or a dotenv loader in app.py).
- Start the Streamlit app:

streamlit run app.py

Open the local URL printed in the terminal.

## Usage
- Upload one or more PDFs from the UI (optional).
- Ask questions. With PDFs uploaded, answers use RAG from a temporary Chroma vectorstore; otherwise the base chat chain responds.
- Use the sidebar to switch/create sessions and inspect brief chat history stats or previews.

## How it works
- PDF ingestion: Pages are extracted with PyPDFLoader, split into overlapping chunks, embedded (sentence-transformers/all-MiniLM-L6-v2), and indexed in Chroma.
- Retrieval: The retriever returns top‑k relevant chunks (k default: 3) to augment the LLM prompt.
- Multi‑session history: Each session maintains its own chat history via LangChain’s message history, enabling persistent context within a session.
- Transparency: Optionally shows “Top retrieved chunks” for the current answer.

## Configuration
- Environment:
  - a.env holds secrets such as GROQ_API_KEY.
- Embeddings:
  - Default: all‑MiniLM‑L6‑v2 (HuggingFace).
- Text splitting:
  - Default: CharacterTextSplitter(chunk_size=1200, chunk_overlap=100); adjust based on document style and context window.
- Retriever:
  - Default top‑k=3; tune for relevance vs. latency.
- Temporary files:
  - PDFs are saved to a temp path for parsing and then removed.

## Project structure
.
├─ app.py           # Streamlit entry point
├─ chains.py        # LLM config and chains: base chat + RAG chain
├─ utils.py         # Session helpers, display utilities, vector DB cleanup
├─ requirements.txt # Dependencies
└─ a.env            # Env vars (e.g., GROQ_API_KEY)

## Tips and tuning
- Documents with long paragraphs: consider overlap 150–200.
- For stricter chunk control: switch to RecursiveCharacterTextSplitter with a token length function.
- Productionizing:
  - Separate a backend API (e.g., FastAPI) from the Streamlit UI.
  - Persist vectorstores (Chroma) on disk or externalize to a managed vector DB.
  - Add authentication, logging, rate limiting, and observability as needed.

## Known limitations
- In‑memory session state; multi‑instance scaling requires external state.
- Local Chroma is ephemeral unless configured to persist.
- PDF parsing quality depends on document structure; scanned PDFs may require OCR.

## License
Add your preferred license (e.g., MIT, Apache‑2.0) here.

## Acknowledgements
Built with Streamlit, LangChain, Chroma, Hugging Face sentence‑transformers, and PyPDF.
