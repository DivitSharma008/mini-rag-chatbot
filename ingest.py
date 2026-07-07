from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv(".env")

loader = PyPDFLoader(r"data\Neural Networks without Calculus.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")
print(docs[0].page_content[:300])

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vs = FAISS.from_documents(chunks, embeddings)
vs.save_local("faiss_index")
print("Vector store rebuilt and saved!")

results = vs.similarity_search("How does a CNN learn using linear algebra?", k=3)
for r in results:
    print(r.page_content[:200], "\n---")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("API_SECRET_KEY"),
    temperature=0,
)

retriever = vs.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

result = qa.invoke({"query": "How does a CNN learn using linear algebra?"})
print(result["result"])

result = qa.invoke({"query": "What's the capital of Mongolia?"})
print(result["result"])