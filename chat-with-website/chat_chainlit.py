import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

def main():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    url = st.text_input("Insert The website URL")

    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        try:
            # Load data from the specified URL
            st.write(f"Loading data from {url}...")
            loader = WebBaseLoader(url)
            data = loader.load()
            st.write("Data loaded successfully.")

            # Split the loaded data
            text_splitter = CharacterTextSplitter(separator='\n', 
                                                  chunk_size=1000, 
                                                  chunk_overlap=40)

            docs = text_splitter.split_documents(data)
            st.write(f"Data split into {len(docs)} documents.")

            # Create Ollama embeddings
            ollama_embeddings = OllamaEmbeddings(model="gemma:2b")

            # Create a Chroma vector database from the documents
            vectordb = Chroma.from_documents(documents=docs, 
                                             embedding=ollama_embeddings,
                                             persist_directory=DB_DIR)

            vectordb.persist()
            st.write("Vector database created and persisted successfully.")

            # Create a retriever from the Chroma vector database
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            # Use a mistral llm from Ollama
            llm = Ollama(model="gemma:2b")

            # Create a RetrievalQA from the model and retriever
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # Run the prompt and return the response
            response = qa(prompt)
            st.write("Query processed successfully.")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Failed to process the URL or query. Please check the URL and try again.")

if __name__ == '__main__':
    main()
