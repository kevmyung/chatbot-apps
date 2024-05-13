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

def render_sidebar() -> Tuple[Dict, Dict]:
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

    return model_info, model_kwargs

def main() -> None:
    model_info, model_kwargs = render_sidebar()

    chat_model = ChatModel(model_info, model_kwargs)
    chain = ConversationChain(
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

    display_chat_messages([]) 
    prompt = st.chat_input()

    if prompt:
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
                    chain, [{"role": "user", "content": formatted_prompt}]
                )
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()