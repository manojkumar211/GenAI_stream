
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import openai,ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.environ.get("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")



def vector_embedding():
    st.session_state.loader=PyPDFLoader("C:/Users/Archana Siripuram/Desktop/attention-is-all-you-need-Paper.pdf")
    st.session_state.document=st.session_state.loader.load()
    st.session_state.splitter=RecursiveCharacterTextSplitter(separators="\n\n",chunk_size=2000,chunk_overlap=500)
    st.session_state.chunks=st.session_state.splitter.split_documents(st.session_state.document[:30])
    st.session_state.embedding=OllamaEmbeddings(model="gemma2:2b")
    st.session_state.vector_store=FAISS(st.session_state.chunks, st.session_state.embedding,docstore=True,index_to_docstore_id=True)
    


st.title("Gen-AI Attention mechanism")

llm=ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)



if st.button("Document Embedding"):
    vector_embedding()
    st.success("Document Embedding is done successfully")

import time

query=st.text_input("Enter your Question regards Attention mechanism :")


if query:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vector_store.as_retriever()
    retriever_chain=create_retrieval_chain([retriever,document_chain])
    start=time.process_time()
    response=retriever_chain.invoke({'input':query})
    print("response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")