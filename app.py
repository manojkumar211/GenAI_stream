from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import streamlit as st


# Load the PDF file

pdf_load=PyPDFLoader("C:/Users/Archana Siripuram/Desktop/attention-is-all-you-need-Paper.pdf")

st.header("My New Generative AI project")
st.logo('C:/Users/Archana Siripuram/Desktop/ge2.png',size="large")
st.image('C:/Users/Archana Siripuram/Desktop/ge1.jpg')
st.chat_input("Enter your Query Here :")
st.button("Submit :")

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