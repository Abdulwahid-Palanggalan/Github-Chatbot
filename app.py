import streamlit as st
from chatbot import load_kb, prepare_kb_embeddings, search_knowledge_base, generate_response

st.set_page_config(page_title="GitHub Knowledge Bot", page_icon="🐙", layout="centered")

st.title("🐙 GitHub Knowledge Bot (100% Free & Local)")
st.markdown("Ask me questions about GitHub features and workflows based strictly on my curated knowledge base. I run entirely offline on your computer!")

# Initialize the local AI model and cache it in Streamlit state 
@st.cache_resource
def initialize_kb():
    with st.spinner("Downloading local AI model and indexing knowledge base... (this might take a few seconds on first run)"):
        kb = load_kb()
        return prepare_kb_embeddings(kb)

try:
    kb_indexed = initialize_kb()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.stop()

# Initialize continuous chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render chat messages from history on UI rerun 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("How do I create a pull request?"):
    
    # Store user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response Processing 
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base locally..."):
            match = search_knowledge_base(prompt, kb_indexed)
            response = generate_response(prompt, match)
            st.markdown(response)
    
    # Store assistant response 
    st.session_state.messages.append({"role": "assistant", "content": response})
