# Mini RAG Chatbot

A compact Retrieval-Augmented Generation (RAG) demo built with Streamlit and LangChain-style components. Upload a PDF and ask natural-language questions about its contents. The app indexes the PDF into vector embeddings (FAISS) and answers questions using a retrieval + LLM pipeline.

GitHub Copilot Chat Assistant

---

## Highlights

- Simple web UI (Streamlit) to upload a PDF and chat with its contents
- Robust PDF loading using PyPDFLoader
- Chunking with overlap to preserve context across splits
- Embeddings via HuggingFace sentence-transformers
- Fast similarity search with FAISS
- Answers produced by a Google Gemini model (via langchain_google_genai) with source citations

---

## Files of note

- `app.py` — Streamlit app: upload, index, and chat (main entrypoint)
- `ingest.py` — Example script to build and save a FAISS index locally (developer/demo)
- `requirements.txt` — Pinned dependencies used during development
- `.gitignore`

---

## Quickstart (local)

1. Clone the repo

   git clone https://github.com/DivitSharma008/mini-rag-chatbot.git
   cd mini-rag-chatbot

2. Create a Python virtual environment and install dependencies

   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

   Note: `requirements.txt` here is comprehensive. For a minimal demo, install at least:
   - streamlit
   - python-dotenv
   - sentence-transformers
   - faiss-cpu
   - pypdf or pypdf2
   - langchain-classic / langchain-community / langchain-huggingface / langchain-google-genai

3. Add environment variables

   Create a `.env` file in the repository root (or set env vars in your system):

   API_SECRET_KEY=your_google_api_key_here

   The app calls `load_dotenv(".env")`, so `.env` will be loaded automatically when present.

4. Run the Streamlit app

   streamlit run app.py

5. Use the UI

   - Upload a PDF (default max file size set to 10 MB in `app.py`)
   - Wait for indexing to finish
   - Ask questions in the chat input; answers and source chunks will display

---

## Configuration and constants

- MAX_FILE_SIZE_MB (in `app.py`) — default 10 MB. Increase if you need bigger uploads but be mindful of memory and browser limits.
- Embedding model — `sentence-transformers/all-MiniLM-L6-v2` (fast and compact). Swap to a larger model for better quality.
- Retrieval k — `k=4` in the app; change to return more or fewer source chunks.
- Caching — embedding model is cached across sessions using `@st.cache_resource`.

---

## How it works (summary)

1. Uploaded PDF is saved to a temporary file (required by some loaders).
2. PyPDFLoader reads pages into Document objects.
3. RecursiveCharacterTextSplitter splits pages into 1000-character chunks with 150-character overlap.
4. HuggingFace sentence-transformers produces embeddings for chunks.
5. FAISS stores embeddings and performs similarity search to find relevant chunks.
6. The selected chunks are passed in a prompt template to the LLM (Gemini via ChatGoogleGenerativeAI) to produce an answer.
7. Answer and source chunks are shown to the user. The prompt explicitly instructs the model to say "I don't know" if the answer is not in the provided context.

Prompt used in `app.py`:

"Use the following context to answer the question. If the answer isn't in the context, say you don't know — don't make things up.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:" 

---

## Developer notes

- `app.py` stores the vectorstore in Streamlit's `session_state` and does not persist it across server restarts. Use `ingest.py` (which calls `vs.save_local("faiss_index")`) if you want to persist the index to disk and reuse it.
- `ingest.py` includes a simple example of building and saving a FAISS index and running sample QA queries. Update the PDF path at the top of that file before running it.
- To switch LLM providers, replace `ChatGoogleGenerativeAI` with your desired LLM wrapper and update the related environment variables.

---

## Performance & costs

- Building embeddings and the FAISS index can be CPU and memory intensive for large PDFs. Consider increasing chunk size or reducing overlap for performance.
- Using managed LLM APIs (e.g., Google Gemini) may incur costs; set `temperature=0` for more deterministic answers and to reduce token usage when appropriate.

---

## Security & privacy

- Uploaded PDFs are written temporarily to disk and deleted immediately after processing. Do not upload sensitive documents to public deployments.
- Do not commit API keys or sensitive config to version control. Use `.env` files locally or secrets managers in production.
- If deploying publicly, protect the app with authentication, HTTPS, and secure secret storage.

---

## Troubleshooting

- PDF fails to load: ensure it's not corrupted and that PyPDFLoader supports it. Try opening the PDF locally.
- Import errors: confirm you're using the same Python version and that dependencies installed correctly in your virtual environment.
- Google API errors: verify your `API_SECRET_KEY` and that the account has access to the Generative AI API.
- Indexing is slow or memory-heavy: reduce chunk count by increasing `chunk_size` or decreasing overlap in the splitter.

---

## Next steps & suggestions

- Persist indexes keyed by file hash to avoid re-indexing large documents.
- Add authentication for any public deployment.
- Add a minimal `requirements-min.txt` with only essential packages for quicker setup.
- Add tests for ingestion and retrieval logic.

---

If you want, I can also:
- Create a Dockerfile to containerize the app
- Add a minimal `requirements-min.txt`
- Add a GitHub Actions workflow to build and run basic checks

