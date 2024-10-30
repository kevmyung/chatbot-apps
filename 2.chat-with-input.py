import streamlit as st
import boto3
from libs.config import load_language_config, load_model_config
from libs.file_utils import handle_file_uploads, process_user_input
from libs.common_utils import handle_language_change, display_ai_response
from typing import List, Dict, Any

MAX_MESSAGE_HISTORY = 20
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png", "txt", "pdf", "csv", "py"]
BEDROCK_REGIONS = ['us-west-2', 'us-east-1', 'ap-northeast-1']

st.set_page_config(page_title='Bedrock AI Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 Bedrock AI Chatbot")

def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": st.session_state.lang_config['init_message']}]

def initialize_session_state():
    if "lang_config" not in st.session_state:
        st.session_state.lang_config = load_language_config("English")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": st.session_state.lang_config['init_message']}]
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def render_sidebar():
    with st.sidebar:
        st.button("New Chat", on_click=new_chat, type="primary")
        st.selectbox('Language 🌎', ['English', 'Korean'], key='language_select', on_change=handle_language_change)

        model_config = load_model_config()
        model_name = st.selectbox(st.session_state.lang_config['model_selection'], list(model_config.keys()), key='model_name')
        model_info = model_config[model_name]

        model_info["region_name"] = st.selectbox(st.session_state.lang_config['region'], BEDROCK_REGIONS, key='bedrock_region')

        system_prompt = st.text_area(st.session_state.lang_config['system_prompt'], value="You're a cool assistant.", key='system_prompt')

        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
        top_p = st.slider('Top P', min_value=0.0, max_value=1.0, value=1.0)

        model_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 4096,
            "system_prompt": system_prompt,
        }

        uploaded_files = st.file_uploader(st.session_state.lang_config['file_selection'], type=SUPPORTED_FILE_TYPES, accept_multiple_files=True, key="file_uploader_key")

    return model_info, model_kwargs, uploaded_files


def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message["role"] == "user":
                user_prompt = message.get("user_prompt_only", message['content'])
                st.markdown(user_prompt)
            else:
                st.markdown(message['content'])


def main():
    initialize_session_state()
    model_info, model_kwargs, uploaded_files = render_sidebar()

    handle_file_uploads(uploaded_files)
    display_chat_messages()

    prompt = st.chat_input()
    process_user_input(prompt, uploaded_files)

    bedrock_client = boto3.client('bedrock-runtime', region_name=model_info['region_name'])
    model_id = model_info['model_id']
    display_ai_response(bedrock_client, model_id, model_kwargs, history_length=MAX_MESSAGE_HISTORY)

if __name__ == "__main__":
    main()