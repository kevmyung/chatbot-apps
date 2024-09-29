import streamlit as st
import boto3
import json
from libs.os_workflow import execute_workflow

BEDROCK_REGIONS = ['us-west-2', 'us-east-1', 'ap-northeast-1']

st.set_page_config(page_title='re:Invent Session Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 re:Invent Session Chatbot")

def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! What would you like to know about re:Invent sessions?"}]

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! What would you like to know about re:Invent sessions?"}]

def render_sidebar():
    with st.sidebar:
        st.button("New Chat", on_click=new_chat, type="primary")
        st.selectbox("Bedrock Region", BEDROCK_REGIONS, key='bedrock_region')

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def display_context(context):
    with st.expander(f"Detailed context for you", expanded=False):
        if isinstance(context, str):
            st.code(context, language="json")
        elif isinstance(context, (dict, list)):
            formatted_context = json.dumps(context, indent=2)
            st.code(formatted_context, language="json")
        else:
            st.code(str(context)) 

def main():
    initialize_session_state()
    render_sidebar()

    display_chat_messages()
    prompt = st.chat_input()
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            progress_expander = st.expander("View Progress", expanded=True)
            progress_container = progress_expander.container()
            with st.spinner('Thinking...'):
                full_response, context = execute_workflow(prompt, progress_container)
            display_context(context)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()