from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

st.title('Langchain Demo With LLAMA2 API')
input_text = st.text_input("Search the topic you want")

llm = Ollama(base_url = 'http://localhost:11434', model="qwen:0.5b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write_stream(llm.stream(input_text))