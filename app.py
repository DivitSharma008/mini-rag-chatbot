import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import os

load_dotenv(".env")
st.set_page_config(page_title="Mini RAG Chatbot", page_icon="📄")
st.title("📄 Mini RAG Chatbot")
st.caption("Upload a PDF and ask questions about it")

PROMPT = PromptTemplate(
    template="""Use the following context to answer the question.
If the answer isn't in the context, say you don't know — don't make things up.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

MAX_FILE_SIZE_MB = 10

@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Cached across ALL sessions/users — the embedding model itself never changes,
    # so there's no reason to reload it per upload
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_index_from_pdf(uploaded_file):
    # Save the uploaded file to a temp path, since PyPDFLoader needs a real file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if len(docs) == 0:
            raise ValueError("No readable text found in this PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # No save_local() here — this index only needs to live in memory
        # for this visitor's session, not persist to disk
        return vectorstore, len(docs), len(chunks)
    finally:
        os.unlink(tmp_path)  # clean up the temp file regardless of success/failure

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f}MB). Please upload a PDF under {MAX_FILE_SIZE_MB}MB.")
        st.stop()

    if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Reading and indexing your PDF..."):
            try:
                vectorstore, num_pages, num_chunks = build_index_from_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.last_file = uploaded_file.name
                st.success(f"Indexed {num_pages} pages ({num_chunks} chunks). Ask away below.")
            except Exception as e:
                st.error(f"Couldn't process this PDF: {e}")
                st.stop()

if "vectorstore" in st.session_state:
    question = st.chat_input("Ask a question about your PDF:")
    if question:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("API_SECRET_KEY"),
            temperature=0,
        )
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )
        with st.spinner("Thinking..."):
            try:
                result = qa.invoke({"query": question})
                st.write(result["result"])
                with st.expander("Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i+1} (page {page})**")
                        st.text(doc.page_content[:300] + "...")
            except Exception as e:
                st.error(f"Something went wrong answering that: {e}")
else:
    st.info("Upload a PDF above to start chatting.")