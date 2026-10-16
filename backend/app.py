import os
from langgraph.graph import StateGraph,START,END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, trim_messages
from typing import TypedDict,Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages.utils import count_tokens_approximately

import sqlite3
from dotenv import load_dotenv

load_dotenv()

conn=sqlite3.connect(database="chat_bot.db",check_same_thread=False)



gem_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",  # Or your chosen model varianta   
    )
groq_llm=ChatGroq(model="openai/gpt-oss-120b")

def call_llm(prompt: str):

    try:
        response = groq_llm.invoke(prompt)
        return response

    except Exception as e:

        if "RESOURCE_EXHAUSTED" in str(e):
            print("Gemini quota exhausted. Falling back to Groq.")

            response = gem_llm.invoke(prompt)
            return response

        raise

message_trimmer = trim_messages(
    max_tokens=4000,                  # Keeps total tokens safely below Groq's free TPM ceilings
    strategy="last",                  # Slices away the OLDEST interactions first
    token_counter=count_tokens_approximately,           # Uses your Groq instance to calculate true model tokens
    include_system=True,              # CRITICAL: Ensures your system prompt is never deleted
    allow_partial=False,              # Prevents clipping sentences or splitting object blocks halfway
    start_on="human",                 # Ensures the trimmed log segment always starts with a human turn
    
)




class ChatState(TypedDict):
    messages:Annotated[BaseMessage,add_messages]



def chat_agent(state:ChatState):
    messages=state["messages"]
    safe_messages = message_trimmer.invoke(messages)
    response = call_llm(safe_messages)
    print("Node call",response)
    # response store state
    return {'messages': [response]}



graph = StateGraph(ChatState)

graph.add_node("chat_agent",chat_agent)
graph.add_edge(START,"chat_agent")
graph.add_edge("chat_agent",END)


checkpoint = SqliteSaver(conn)

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_thread():
    all_threads=set()
    for chek in checkpoint.list(None):
        all_threads.add(chek.config["configurable"]["thread_id"])
    return list(all_threads)