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
from libs.chat_utils import StreamHandler, display_chat_messages, display_pdf_images, langchain_messages_format
from libs.file_utils import faiss_preprocess_document, faiss_reset_on_click

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
    return conversation.invoke({"input": input}, {"callbacks": [StreamHandler(st.empty())]})

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

        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 200,
            "max_tokens": 4096,
            "system": """You are a helpful assistant that answers users' questions based on the context. 
            Offer kind and accurate responses based on the given context. If the answer is not in the provided context, respond that you do not know."""
        }

        # File Uploader
        uploaded_files = st.file_uploader(
            lang_config['file_selection'],
            type=["pdf"],
            accept_multiple_files=True,
            key="file_uploader_key"
        )

        st.button(
            lang_config['clean_knowledge'],
            on_click=faiss_reset_on_click
        )
        st.session_state['clean_kb_message'] = lang_config['clean_kb_message']

    return model_info, model_kwargs, uploaded_files

def main() -> None:
    model_info, model_kwargs, uploaded_files = render_sidebar()

    chat_model = ChatModel(model_info, model_kwargs)
    memory = ConversationBufferWindowMemory(k=10, ai_prefix="Assistant", chat_memory=StreamlitChatMessageHistory(), return_messages=True)
    chain = ConversationChain(
        llm=chat_model.llm,
        verbose=True,
        memory=memory,
        prompt=CLAUDE_PROMPT,
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [INIT_MESSAGE]

    if "vector_empty" not in st.session_state:
        st.session_state["vector_empty"] = True

    display_chat_messages(uploaded_files)

    prompt = st.chat_input()

    retriever = faiss_preprocess_document(uploaded_files, chat_model, lang_config['upload_message'])
    if prompt:
        context_text = ""
        with st.chat_message("user"):
            st.markdown(prompt)

            context_text = retriever.get_relevant_documents(prompt)

            prompt_new = f"Here's some context for you. However, do not mention this context unless it is directly relevant to the user's question. It is essential to deliver an answer that precisely addresses the user's needs \n<context>\n{context_text}</context>\n\n{prompt}\n\n"

            formatted_prompt = chat_model.format_prompt(prompt_new)
            st.session_state.messages.append({"role": "user", "content": formatted_prompt})
            
            st.session_state["langchain_messages"] = langchain_messages_format(
                st.session_state["langchain_messages"]
            )

        with st.chat_message("assistant"):
            response = generate_response(
                chain, [{"role": "user", "content": formatted_prompt}]
            )
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            
            if 'image_paths' in st.session_state:
                for img in st.session_state['image_paths']:
                    display_pdf_images(img)

if __name__ == "__main__":
    main()
