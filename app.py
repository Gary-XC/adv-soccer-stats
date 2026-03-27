import streamlit as st
import requests

st.set_page_config(page_title="Soccer AI Scout", page_icon="⚽", layout="wide")

# --- SESSION STATE INITIALIZATION ---
# This keeps the chat history visible on the screen
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("⚽ Advanced Soccer Analytics: AI Scout")
st.markdown("Query custom engineered European football metrics using natural language.")

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT ---
if prompt := st.chat_input("Ask about player usage, TFE, or stats..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing dataset..."):
            try:
                # We send the question to our FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": prompt, "session_id": "default_user"},
                )

                if response.status_code == 200:
                    answer = response.json().get("answer")
                    st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- SIDEBAR: PROJECT CONTEXT ---
with st.sidebar:
    st.header("Engine Specs")
    st.info(
        """
    - **Model:** Llama 3.1 (Local)
    - **Memory:** Window Buffer (k=5)
    - **Metrics:** Attacking Usage, TFE, Possession USG
    """
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
