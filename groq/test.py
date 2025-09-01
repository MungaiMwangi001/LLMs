import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables and initialize the client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Groq Chatbot",
    page_icon=":rocket:",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title(" Groq Chatbot with Streamlit")

# Sidebar for model selection
st.sidebar.header(" Settings")
models = client.models.list()
model_list = [m.id for m in models.data]
selected_model = st.sidebar.selectbox("Select a model", model_list, index=0)

# Clear chat button
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Chat cleared! How can I help you now?"}]

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I'm your Groq AI assistant. How can I help you today?"}]

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Get response from Groq
    with st.spinner("Thinking..."):
        chat_completion = client.chat.completions.create(
            model=selected_model,
            messages=st.session_state["messages"],
            temperature=0.7
        )
        response = chat_completion.choices[0].message.content

    # Add assistant response
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
