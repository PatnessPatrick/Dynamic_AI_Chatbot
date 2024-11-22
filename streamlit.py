import streamlit as st
from datetime import datetime
from main import ConversationManager

# Initialize Streamlit app
st.title("AI Chatbot: Interactive Conversation Manager")

# Initialize Session State and Chat Manager
if "conversation_manager" not in st.session_state:
    st.session_state["conversation_manager"] = ConversationManager(
        history_file=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    st.session_state["conversation_history"] = []
    st.session_state["last_input"] = ""  # Initialize 'last_input'

# Alias for convenience
conversation_manager = st.session_state["conversation_manager"]

# Sidebar Widgets for Customization
st.sidebar.header("Chat Settings")
chat_temperature = st.sidebar.slider("Conversation Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens per Response", 50, 1000, 512, 50)
token_budget = st.sidebar.slider("Max Token Budget for Conversation", 1000, 8000, 4000, 500)

persona = st.sidebar.selectbox(
    "Choose Chatbot Persona",
    ["Sassy Assistant", "Angry Assistant", "Thoughtful Assistant", "Custom"]
)

# Handle persona selection
if persona == "Custom":
    custom_persona = st.sidebar.text_input("Enter Custom Persona Message")
    if st.sidebar.button("Set Custom Persona"):
        if custom_persona.strip():
            conversation_manager.set_custom_system_message(custom_persona)
            st.success("Custom persona set!")
        else:
            st.warning("Custom persona message cannot be empty.")
else:
    persona_map = {
        "Sassy Assistant": "sassy_assistant",
        "Angry Assistant": "angry_assistant",
        "Thoughtful Assistant": "thoughtful_assistant",
    }
    conversation_manager.set_persona(persona_map[persona])

# Reset Chat History
if st.sidebar.button("Reset Conversation History"):
    conversation_manager.reset_conversation_history()
    st.session_state["conversation_history"] = []
    st.success("Conversation history reset!")

# Chat Functionality
user_input = st.chat_input("Type your message here...")
if user_input:
    if user_input != st.session_state["last_input"]:  # Ensure input is processed only once
        st.session_state["last_input"] = user_input
        
        # Generate chatbot response
        response = conversation_manager.chat_completion(
            prompt=user_input,
            temperature=chat_temperature,
            max_tokens=max_tokens,
        )
        
        # Append new messages to conversation history
        if response:
            st.session_state["conversation_history"].append({"role": "user", "content": user_input})
            st.session_state["conversation_history"].append({"role": "assistant", "content": response})

# Display Conversation History
for message in st.session_state["conversation_history"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])