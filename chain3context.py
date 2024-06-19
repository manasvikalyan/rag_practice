from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the app
@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="qwen:0.5b")  # Set the model to qwen:0.5b
    cl.user_session.set("model", model)
    cl.user_session.set("message_history", [])  # Initialize an empty message history

# Resume a conversation
@cl.on_chat_resume
async def on_chat_resume():
    model = Ollama(model="qwen:0.5b")  # Reinitialize the model
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
    model = cl.user_session.get("model")  # Get the model
    history = cl.user_session.get("message_history")  # Get the message history

    # Add the current message to the message history
    history.append({"role": "human", "content": message.content})
    
    # Prepare the context from history
    context = " ".join([entry["content"] for entry in history])
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a good assistant."),
            ("human", context)
        ]
    )
    runnable = prompt | model | StrOutputParser()
    
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

# Main entry point
if __name__ == "__main__":
    cl.run(persist=True)  # Enable data persistence
