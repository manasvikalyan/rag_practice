from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Authentication Callback
@cl.header_auth_callback
def header_auth_callback(headers: dict) -> cl.User:
    # Verify authentication header
    if headers.get("Authorization") == f"Bearer {os.getenv('AUTH_TOKEN')}":
        return cl.User(identifier="user_id", metadata={"role": "user"})
    else:
        return None

# Initialize the app
@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="qwen:0.5b")  # Set the model to qwen:0.5b
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a good assistant.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("message_history", [])  # Initialize an empty message history

# Resume a conversation
@cl.on_chat_resume
async def on_chat_resume():
    runnable = cl.user_session.get("runnable")  # type: Runnable
    history = cl.user_session.get("message_history")  # Get the persisted message history
    if runnable:
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
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    
    # Add the current message to the message history
    history = cl.user_session.get("message_history")
    history.append({"role": "human", "content": message.content})
    cl.user_session.set("message_history", history)

# Main entry point
if __name__ == "__main__":
    cl.run(persist=True, auth=True)  # Enable data persistence and authentication
