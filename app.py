# File: app.py
import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load API key from .env
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

st.title("Research Paper Buddy")

if "vectors" not in st.session_state:
    loader = WebBaseLoader("https://arxiv.org")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs[:30])
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.vectors = vectorstore

retriever = st.session_state.vectors.as_retriever()
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below:
<context>
{context}
</context>
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

query = st.text_input("Ask a question from arXiv docs")

if query:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": query})
    st.write("Answer:", response['answer'])
    st.write("---")
    st.subheader("Retrieved Context")
    for doc in response['context']:
        st.write(doc.page_content)
        st.write("---")
