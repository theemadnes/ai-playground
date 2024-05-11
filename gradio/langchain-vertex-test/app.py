import gradio as gr
#from dotenv import load_dotenv
#import os
# Import Langchain chat model for Vertex AI
#from langchain.memory.buffer import ConversationBufferMemory
#from langchain_community import ConversationBufferMemory
from langchain_google_vertexai import ChatVertexAI
#from langchain.chains import ConversationChain

from langchain.schema import AIMessage, HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Chat
chat = ChatVertexAI()
chat_memory = ConversationBufferMemory()

# Gradio ChatInterface Function
def predict(message, history):
    conversation = ConversationChain(
        llm=chat, verbose=True, memory=chat_memory
    )
    return conversation.predict(input=message)

gr.ChatInterface(predict).launch(share=True)