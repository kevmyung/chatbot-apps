import streamlit as st
import random
from typing import Dict, Tuple, List, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from libs.config import load_model_config, load_language_config
from libs.opensearch import OpenSearchClient, get_opensearch_retriever
from libs.models import ChatModel
from libs.chat_utils import StreamHandler, display_chat_messages, langchain_messages_format
from libs.file_utils import opensearch_preprocess_document, opensearch_reset_on_click

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

def render_sidebar() -> Tuple[Dict, Dict, List[st.runtime.uploaded_file_manager.UploadedFile], List]:
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
            accept_multiple_files=False,
            key="file_uploader_key"
        )

    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.button(
            lang_config['init_knowledge'],
            on_click=opensearch_reset_on_click
        )
        st.session_state['init_kb_message'] = lang_config['init_kb_message']

    with col2:
        semantic_weight = st.slider(lang_config['hybrid_weight'], 0.0, 1.0, 0.51, 0.01, 
                                    help=lang_config['hybrid_desc'], 
                                    key="semantic_weight_sidebar")
        ensemble = [semantic_weight, 1 - semantic_weight]
        st.session_state["ensemble"] = ensemble

    return model_info, model_kwargs, uploaded_files, ensemble


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [INIT_MESSAGE]

    if "vector_empty" not in st.session_state:
        st.session_state["vector_empty"] = True

def main() -> None:
    #init_session_state()

    model_info, model_kwargs, uploaded_files, ensemble = render_sidebar()

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

    if "os_client" not in st.session_state:
        os_client = OpenSearchClient(llm = chat_model.llm, emb = chat_model.emb)
        st.session_state["os_client"] = os_client
    else:
        os_client = st.session_state["os_client"]
    opensearch_preprocess_document(uploaded_files, chat_model, os_client, lang_config['upload_message'])
    os_retriever = get_opensearch_retriever(os_client)

    is_vector_empty = st.session_state["vector_empty"]
    if prompt:
        context_text = ""
        if not is_vector_empty == True:
            context_text = os_retriever._get_relevant_documents(query = prompt, ensemble = ensemble)

        prompt_new = f"Here's some context for you. However, do not mention this context unless it is directly relevant to the user's question. It is essential to deliver an answer that precisely addresses the user's needs \n<context>\n{context_text}</context>\n\n{prompt}\n\n"

        formatted_prompt = chat_model.format_prompt(prompt_new)
        st.session_state.messages.append({"role": "user", "content": formatted_prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state["langchain_messages"] = langchain_messages_format(
            st.session_state["langchain_messages"]
        )

        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = generate_response(
                    chain, [{"role": "user", "content": formatted_prompt}]
                )
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
