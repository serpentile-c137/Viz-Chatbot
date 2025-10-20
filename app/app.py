# app.py
import streamlit as st
from main import ask_agent_with_plot

st.set_page_config(page_title="QAV-Agent", layout="centered")
st.title("Agentic Chatbot for Data Q&A and Visualization")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history using st.chat_message
for message in st.session_state["chat_history"]:
    with st.chat_message(message["type"]):
        st.markdown(message["content"])
        if message.get("plot"):
            st.image(message["plot"], use_column_width=True)

# Input for user query
if query := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state["chat_history"].append({"type": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process the query
    with st.spinner("Agent is thinking and calling tools..."):
        text, plot = ask_agent_with_plot(query)
    
    # Add agent response to chat history
    response = {"type": "assistant", "content": text}
    if plot:
        response["plot"] = plot
    st.session_state["chat_history"].append(response)
    
    # Display agent response
    with st.chat_message("assistant"):
        st.markdown(text)
        if plot:
            st.image(plot, use_column_width=True)