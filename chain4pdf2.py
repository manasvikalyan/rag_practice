# main.py

import os
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")

# Set up RetrievalQA model
rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")

def load_model():
    llm = Ollama(
        model="qwen:0.5b",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_mistral},
        return_source_documents=True,
    )
    return qa_chain

def qa_bot():
    llm = load_model()
    DB_PATH = DB_DIR
    vectorstore = Chroma(
        persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="qwen:0.5b")
    )
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Ollama (qwen:0.5b model) and LangChain."
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    text_elements = []  # List[cl.Text]

    source_documents = res.get("source_documents", [])

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )

    await cl.Message(content=answer, elements=text_elements).send()

if __name__ == "__main__":
    cl.run(persist=True)
