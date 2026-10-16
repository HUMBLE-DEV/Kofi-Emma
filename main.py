import uuid
from dotenv import load_dotenv
load_dotenv()

from backend.app import chatbot, get_all_thread 
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import streamlit as st

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)

def reset_chat():
    st.session_state["message_history"]=[]
    st.session_state["thread_id"]=generate_thread_id()
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    stat=chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }}
    )
    return stat.values.get("messages",[])

st.title("Agentic Chatbot with LangGraph")  

#CONFIG = {'configurable': {'thread_id': st.session_state["thread_id"]}}

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]


if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"]=get_all_thread()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=generate_thread_id()

add_thread(st.session_state["thread_id"])



# ========================= Sidebar threading feature =========================
st.sidebar.title("Conversations")
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_thread"][::-1]:
    if st.sidebar.button(str(thread_id),key=thread_id):
        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message,HumanMessage):
                role="user"
            elif isinstance(message,AIMessage):
                role='assistant'

            else:
                continue
            temp_messages.append({
                "role":role,
                "content":message.content
            })
        st.session_state["message_history"]=temp_messages
        #st.rerun()


for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.markdown(message["content"])



user_input=st.chat_input("Type Here")

def stream_response(user_input,active_thread):
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": active_thread}},
        stream_mode="messages"
    ):
        content = message_chunk.content
        if isinstance(content,list):
            for item in content:
                if isinstance(item, dict):
                    yield item.get("text", "")
        else:
            print("AM not a list OOO am a ----------->>>>" ,type(content))
            yield content



if user_input:
    st.session_state["message_history"].append(
        {'role': 'user', 'content': user_input}
                                                )
    with st.chat_message("assistant"):
        ai_message=st.write_stream(
            stream_response(user_input,st.session_state["thread_id"]))

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})