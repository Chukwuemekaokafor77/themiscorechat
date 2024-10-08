import streamlit as st
import os
import requests
import json

# Set API keys and URLs from environment variables
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

if not CLAUDE_API_KEY:
    st.error("Claude API key is not set. Please ensure the CLAUDE_API_KEY environment variable is correctly configured.")

# Function to call Claude API
def call_claude_api(messages):
    try:
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "messages": messages,
            "system": get_specialized_system_prompt()
        }
        
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            st.error(f"Error from Claude API: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        st.error(f"Claude API error: {e}")
        return None

# Function to generate system prompt for legal expertise
def get_specialized_system_prompt():
    return """You are a specialized legal assistant with expertise in contract law, civil litigation, and regulatory compliance. 
    Your responses should:
    1. Use precise legal terminology
    2. Reference relevant legal principles and precedents when applicable
    3. Maintain a formal, professional tone
    4. Clarify any ambiguities in legal language
    5. Highlight potential legal implications or considerations

    When analyzing documents or answering questions:
    - Consider jurisdictional context
    - Identify potential legal issues or red flags
    - Provide balanced analysis of legal positions
    - Clarify limitations of general legal information vs. legal advice."""

# Few-shot examples to guide Claude's responses
def create_few_shot_examples():
    return [
        {
            "role": "user",
            "content": "What are the key elements of a valid contract?"
        },
        {
            "role": "assistant",
            "content": "A valid contract requires several essential elements under common law principles:\n\n1. Offer: A clear proposal to enter into an agreement\n2. Acceptance: Unequivocal agreement to the terms of the offer\n3. Consideration: Exchange of something of value between parties\n4. Intention to create legal relations: Parties must intend to be legally bound\n5. Capacity: Parties must have legal capacity to contract\n6. Legality: The purpose and terms must be lawful\n\nMissing any of these elements may render the contract void, voidable, or unenforceable."
        }
    ]

# Function to process user query and interact with Claude
def chat_with_claude(user_query, conversation_history):
    # Prepare conversation with few-shot examples and history
    messages = create_few_shot_examples()
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_query})
    
    # Call Claude API
    return call_claude_api(messages)

# Streamlit UI setup
st.set_page_config(page_title="Legal Assistant Chatbot", page_icon="⚖️")

st.title("🤖 Legal Assistant Chatbot")
st.markdown("""
This AI-powered legal assistant can help you with:
- Contract law questions
- Civil litigation information
- Regulatory compliance guidance

*Note: This chatbot provides general legal information, not legal advice.*
""")

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Create a container for the chat history
chat_container = st.container()

# Create a container for the user input at the bottom of the page
with st.container():
    user_query = st.text_input("Your legal question:", key="user_input")
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send")
    with col2:
        clear_button = st.button("Clear Conversation")

# Process user input
if send_button and user_query:
    with st.spinner('Thinking...'):
        chatbot_response = chat_with_claude(user_query, st.session_state.conversation_history)
        
        if chatbot_response:
            # Update conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_query})
            st.session_state.conversation_history.append({"role": "assistant", "content": chatbot_response})

# Clear conversation if clear button is clicked
if clear_button:
    st.session_state.conversation_history = []
    st.experimental_rerun()

# Display conversation history
with chat_container:
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f"🧑‍💼 **You**: {message['content']}")
        else:
            st.markdown(f"🤖 **Assistant**: {message['content']}")

# Add a divider between chat history and input
st.markdown("---")
