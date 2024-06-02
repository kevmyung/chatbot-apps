import streamlit as st
import random
import os
from typing import Dict, Tuple, List, Union
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from libs.db_utils import DatabaseClient
from libs.config import load_model_config, load_language_config
from libs.models import ChatModel
from libs.opensearch import OpenSearchClient, get_opensearch_retriever
from libs.chat_utils import display_chat_messages
from libs.file_utils import sample_query_indexing, process_schema_description

st.set_page_config(page_title='Bedrock AI Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 Bedrock AI Chatbot")

lang_config = {}
INDEX_NAME = 'example_queries'
INIT_MESSAGE = {}

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

def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []

def render_sidebar() -> Tuple[str, Dict, Dict, Dict]:
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
            "max_tokens": 20480,
            "system": """
            You are a helpful assistant for answering questions in {language}. 
            Firstly, explain the process that led to the final answer. 
            If you used the SQL tools for resolving the user's question, provide the detailed answer to the user's question with numbers and the used SQL queries within a Markdown code block.""".format(language=language)
         }

        # Database Selector
        database_selection = st.selectbox(
            lang_config['database'],
            ('SQLite (Sample)', 'MySQL', 'PostgreSQL', 'Redshift', 'SQLite', 'Presto', 'Oracle')
        )

        if database_selection != "SQLite (Sample)":
            database_dialect = database_selection
            database_uri = st.text_input("Database URI", value="", placeholder="dbtype://user:pass@hostname:port/dbname")
            if not database_uri:
                st.info(lang_config['database_uri'])
                st.stop()
        else:
            database_dialect = "SQLite"
            database_uri = "sqlite:///Chinook.db"

        with st.sidebar:
            add_schema_desc = st.checkbox(lang_config['schema_desc'], value=False)
            schema_file = ""
            if add_schema_desc:
                schema_file = st.text_input(lang_config['schema_file'], value="libs/default-schema.json")                
                process_schema_description(schema_file, model_info["region_name"], lang_config)

        with st.sidebar:
            allow_query_exec = st.checkbox(lang_config['query_exec'], value=False)

        database_region = model_info["region_name"] # change if not same
        database_config = {
            "dialect": database_dialect,
            "uri": database_uri,
            "add_schema_desc": add_schema_desc,
            "allow_query_exec": allow_query_exec,
            "region": database_region
        }

    return model_info, model_kwargs, database_config


def main() -> None:
    model_info, model_kwargs, database_config = render_sidebar()
    chat_model = ChatModel(model_info, model_kwargs)
    with st.sidebar:
        enable_rag_query = st.checkbox(lang_config['rag_query'], value=False)
        if enable_rag_query:
            os_client = OpenSearchClient(emb=chat_model.emb, index_name=INDEX_NAME, mapping_name='mappings-sql', vector="input_v", text="input", output=["input", "query"])
            sample_query_indexing(os_client, lang_config)
        database_config['enable_rag_query'] = enable_rag_query

    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE] 
            
    display_chat_messages([])

    prompt = st.chat_input(placeholder=lang_config['example_msg'])

    if prompt:        
        st.session_state.messages.append({"role": "user", "content": prompt})         
        st.chat_message("user").write(prompt)

        db_client = DatabaseClient(chat_model.llm, database_config)
        qa_examples = ""
        if enable_rag_query:
            os_retriever = get_opensearch_retriever(os_client)
            samples = os_retriever.invoke(prompt, ensemble = [0.51, 0.49])
        
        with st.chat_message("assistant"):
            callback = StreamlitCallbackHandler(st.container())
            response = db_client.sql_executor.invoke({"question":prompt, "dialect":db_client.dialect, "samples":samples, "chat_history": st.session_state.messages}, config={"callbacks": [callback]})
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            st.write(response['output'])

if __name__ == "__main__":
    main()