import os
import streamlit as st
import random
from typing import Dict, Tuple, List, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from libs.config import load_model_config, load_language_config
from libs.models import ChatModel
from libs.chat_utils import StreamHandler, display_chat_messages, langchain_messages_format
from libs.file_utils import process_uploaded_files

st.set_page_config(page_title='Bedrock AI Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 Bedrock AI Chatbot")
lang_config = {}

INIT_MESSAGE = {}
CLAUDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ]
)

def set_init_message(init_message):
    global INIT_MESSAGE
    INIT_MESSAGE = {
        "role": "assistant",
        "content": init_message
    }

def handle_language_change():
    global lang_config, INIT_MESSAGE
    lang_config = load_language_config(st.session_state['language_select'])
    set_init_message(lang_config['init_message'])
    new_chat()

def generate_response(conversation: ConversationChain, input: Union[str, List[dict]]) -> str:
    return conversation.invoke(
        {"input": input}, {"callbacks": [StreamHandler(st.empty())]}
    )

def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []

def render_sidebar() -> Tuple[Dict, Dict, List[st.runtime.uploaded_file_manager.UploadedFile]]:
    st.sidebar.button("New Chat", on_click=new_chat, type="primary")
    with st.sidebar:
        # Language
        global lang_config
        language = st.selectbox(
            'Language 🌎',
            ['Korean', 'English'],
            key='language_select',
            on_change=handle_language_change
        )
        lang_config = load_language_config(language)
        set_init_message(lang_config['init_message'])

        # Model
        model_config = load_model_config()
        model_name_select = st.selectbox(
            lang_config['model_selection'],
            list(model_config.keys()),
            key='model_name',
        )
        model_info = model_config[model_name_select]

        # Region
        model_info["region_name"] = st.selectbox(
            lang_config['region'],
            ['us-east-1', 'us-west-2', 'ap-northeast-1'],
            key='bedrock_region',
        )

        # System Prompt
        system_prompt = st.text_area(
            lang_config['system_prompt'],
            value="You're a cool assistant, love to respond with emoji.",
            key='system_prompt',
        )

        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 200,
            "max_tokens": 4096,
        }
        model_kwargs["system"] = system_prompt

        # File Uploader
        uploaded_files = st.file_uploader(
            lang_config['file_selection'],
            type=["jpg", "jpeg", "png", "txt", "pdf", "csv", "py"],
            accept_multiple_files=True,
            key="file_uploader_key"
        )

    return model_info, model_kwargs, uploaded_files

def main() -> None:

    model_info, model_kwargs, uploaded_files = render_sidebar()

    chat_model = ChatModel(model_info, model_kwargs)
    conv_chain = ConversationChain(
        llm=chat_model.llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=10, ai_prefix="Assistant",
            chat_memory=StreamlitChatMessageHistory(),
            return_messages=True,
        ),
        prompt=CLAUDE_PROMPT,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

    # Display message history
    display_chat_messages(uploaded_files)

    # User chat
    prompt = st.chat_input()

    # Images already processed
    message_images_list = [
        image_id
        for message in st.session_state.messages
        if message["role"] == "user"
        and "images" in message
        and message["images"]
        for image_id in message["images"]
    ]

    uploaded_file_ids = []
    if uploaded_files and len(message_images_list) < len(uploaded_files):
        with st.chat_message("user"):
            content_files = process_uploaded_files(
                uploaded_files, message_images_list, uploaded_file_ids
            )

            if prompt:
                context_text = ""
                context_image = []
                for content_file in content_files:
                    if content_file['type'] == 'text':
                        context_text += content_file['text'] + "\n\n"
                    else:
                        context_image.append(content_file)

                if context_text != "":
                    prompt_new = f"Here is some context for you: \n<context>\n{context_text}</context>\n\n{prompt}"
                else:
                    prompt_new = prompt

                formatted_prompt = chat_model.format_prompt(prompt_new) + context_image
                st.session_state.messages.append(
                    {"role": "user", "content": formatted_prompt, "images": uploaded_file_ids}
                )
                st.markdown(prompt)

    elif prompt:
        formatted_prompt = chat_model.format_prompt(prompt)
        st.session_state.messages.append({"role": "user", "content": formatted_prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    st.session_state["langchain_messages"] = langchain_messages_format(
        st.session_state["langchain_messages"]
    )

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = generate_response(
                conv_chain, [{"role": "user", "content": formatted_prompt}]
            )
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()