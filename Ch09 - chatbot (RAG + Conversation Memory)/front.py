import streamlit as st 
import lib as glib

index = glib.get_index()

st.set_page_config(page_title="Chatbot")
st.title("Chatbot") 

if 'memory' not in st.session_state:
    # lib.py 파일로부터 get_memory() 함수를 호출.
    st.session_state.memory = glib.get_memory() 

if 'chat_history' not in st.session_state: 
    st.session_state.chat_history = [] 

for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"])

input_text = st.chat_input("Chat with your bot here") 

if input_text:  
    with st.chat_message("user"):
        st.markdown(input_text) 
    st.session_state.chat_history.append({"role":"user", "text":input_text})  

    # lib.py 파일로부터 get_chat_response() 함수를 호출.
    chat_response = glib.get_rag_chat_response(input_text=input_text, memory=st.session_state.memory, index=index)  
    with st.chat_message("assistant"): 
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role":"assistant", "text":chat_response})