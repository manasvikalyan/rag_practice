import os
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import chainlit as cl

# Import necessary modules from langchain_community and litters.base
from langchain_community.llms import Ollama
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.docstore.document import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableConfig

# Import text splitter classes
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Initialize Ollama model and message history
@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="qwen:0.5b")  # Initialize Ollama model
    cl.user_session.set("model", model)
    cl.user_session.set("message_history", [])  # Initialize an empty message history
    cl.user_session.set("pdf_text", "")  # Initialize PDF text storage
    cl.user_session.set("pdf_sections", {})  # Initialize PDF sections storage

# Resume a conversation
@cl.on_chat_resume
async def on_chat_resume():
    model = Ollama(model="qwen:0.5b")  # Reinitialize the Ollama model
    cl.user_session.set("model", model)
    history = cl.user_session.get("message_history")  # Get the persisted message history
    if history:
        for entry in history:
            role = entry.get("role", "system")  # Default to system role if not specified
            content = entry.get("content", "")
            if role == "system":
                await cl.Message(content=content).send()
            elif role == "human":
                await cl.Message(content=content).send()

# Process incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    model = cl.user_session.get("model")  # Get the Ollama model
    history = cl.user_session.get("message_history")  # Get the message history

    # Add the current message to the message history
    history.append({"role": "human", "content": message.content})
    
    # Prepare the context from history
    context = " ".join([entry["content"] for entry in history])
    
    # Create a prompt from messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a good assistant."),
        ("human", context)
    ])
    
    # Create a runnable pipeline with Ollama model
    runnable = prompt | model | StrOutputParser()
    
    # Send the processed message
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    
    # Add the model's response to the message history
    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("message_history", history)

# Function to handle file upload and processing
async def process_file(file_path: str, file_name: str):
    msg = cl.Message(content=f"Processing `{file_name}`...", disable_feedback=True)
    await msg.send()

    # Extract text from the PDF file
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    # Store extracted text in user session
    cl.user_session.set("pdf_text", text)

    # Split the text into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Ollama model
    chain = ConversationalRetrievalChain.from_llm(
        cl.user_session.get("model"),
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file_name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

# Handle file upload through a chat message
@cl.on_message
async def handle_file_upload(message: cl.Message):
    if message.content.startswith("upload:"):
        file_path = message.content.split("upload:")[1].strip()
        file_name = os.path.basename(file_path)
        await process_file(file_path, file_name)
        await cl.Message(content=f"File `{file_name}` uploaded and processed.").send()
    else:
        # Check if there's a chain available
        chain = cl.user_session.get("chain")
        if chain:
            result = await chain.acall({"question": message.content})
            response = result["answer"]
            await cl.Message(content=response).send()
        else:
            await on_message(message)

# Main entry point
if __name__ == "__main__":
    # Run the chainlit application with data persistence
    cl.run(persist=True)
