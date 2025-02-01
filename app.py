from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retriever_chain


# Load the PDF file

pdf_load=PyPDFLoader("C:/Users/Archana Siripuram/Desktop/attention-is-all-you-need-Paper.pdf")

st.header("My New Generative AI project")
st.logo('C:/Users/Archana Siripuram/Desktop/ge2.png',size="large")
st.image('C:/Users/Archana Siripuram/Desktop/ge1.jpg')
st.chat_input("Enter your Query Here :")

class pdf_loader:
    pdf_doc=pdf_load.load()

    def __init__(self, pdf_doc):
        self.pdf_doc=pdf_doc

    def pdf_load_doc(self):
        return self.pdf_doc
        

print(pdf_loader.pdf_doc[0].page_content)

# Text splitter

class text_splitter:
    cts_gen=CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=500
)
    cts_doc=cts_gen.split_documents(pdf_loader.pdf_doc)
    def __init__(self,cts_gen,cts_doc):
        self.cts_gen=cts_gen
        self.cts_doc=cts_doc

    def text_split_gen(self):
        return self.cts_gen
    def text_split_doc(self):
        return self.cts_doc


print(text_splitter.cts_doc[0].page_content)
print("##"*50)
print(text_splitter.cts_doc[1].page_content)

# Embedded Techniques:

class oll_embeddings:
    oll_emd=OllamaEmbeddings(model="gemma2:2b")
    def __init__(self,oll_emd):
        self.oll_emd=oll_emd

    def embed_open(self):
        return self.oll_emd
    

# Vector Store Database:

class vector_store_faiss:
    faiss_vector=FAISS(text_splitter.cts_doc, oll_embeddings.oll_emd,docstore=True,index_to_docstore_id=True)
    retriever_vec=faiss_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def __init__(self,faiss_vector,retriever_vec):
        self.faiss_vector=faiss_vector
        self.retriever_vec=retriever_vec

    def vector_search(self):
        return self.faiss_vector
    def vector_retriever(self):
        return self.retriever_vec
    
# vector_store_faiss.faiss_vector.save_local("faiss_vectoreDB")







