import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

def main():
    st.title("Chat with Website")
    st.write("This is a simple chatbot that can answer questions based on the content of a website.")
    url=st.text_input("Enter the URL of the website you want to chat with:")
    prompt=st.text_input("Enter your question:")
    if st.button('Submit question', type='primary'):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        loader=WebBaseLoader(url)
        data=loader.load()

        text_spiltter=CharacterTextSplitter(separator='\n',chunk_size=1000, chunk_overlap=40)
        docs=text_spiltter.split_documents(data)

        ollama_emebeddings=OllamaEmbeddings(model="gemma:2b")
        vector_db=Chroma.from_documents(documents=docs,embedding=ollama_emebeddings,persist_directory=DB_DIR)

        vector_db.persist()

        retriever=vector_db.as_retriever(search_kwargs={"k":5})

        llm=Ollama(model="gemma:2b")

        qa=RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever)

        response=qa(prompt)
        st.write(response)

if __name__ == "__main__":
    main()

