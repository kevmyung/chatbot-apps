import streamlit as st
import json
from typing import Dict, Tuple, List, Union
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from libs.db_utils import DatabaseClient
from libs.config import load_model_config, load_language_config
from libs.models import ChatModel
from libs.opensearch import OpenSearchClient
from libs.chat_utils import display_chat_messages
from libs.file_utils import sample_query_indexing, schema_desc_indexing

st.set_page_config(page_title='Bedrock AI Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 Bedrock AI Chatbot")

lang_config = {}
INIT_MESSAGE = {"role": "assistant", "content": ""}

def set_init_message(init_message: str) -> None:
    INIT_MESSAGE["content"] = init_message

def handle_language_change() -> None:
    global lang_config
    lang_config = load_language_config(st.session_state['language_select'])
    set_init_message(lang_config['init_message'])
    new_chat()

def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []

def render_sidebar() -> Tuple[Dict, Dict, Dict]:
    st.sidebar.button("New Chat", on_click=new_chat, type="primary")
    global lang_config
    language = st.sidebar.selectbox(
        'Language 🌎',
        ['Korean', 'English'],
        key='language_select',
        on_change=handle_language_change
    )
    lang_config = load_language_config(language)
    set_init_message(lang_config['init_message'])

    model_config = load_model_config()
    model_name_select = st.sidebar.selectbox(
        lang_config['model_selection'],
        list(model_config.keys()),
        key='model_name',
    )
    model_info = model_config[model_name_select]

    model_info["region_name"] = st.sidebar.selectbox(
        lang_config['region'],
        ['us-east-1', 'us-west-2', 'ap-northeast-1'],
        key='bedrock_region',
    )

    model_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 200,
        "max_tokens": 20480,
        "system": f"""
        You are a helpful assistant for answering questions in {language}. 
        Explain the process that led to the final answer."""
    }

    database_selection = st.sidebar.selectbox(
        lang_config['database'],
        ('SQLite (Sample)', 'MySQL', 'PostgreSQL', 'Redshift', 'SQLite', 'Presto', 'Oracle')
    )

    if database_selection != "SQLite (Sample)":
        database_dialect = database_selection
        database_uri = st.sidebar.text_input("Database URI", value="", placeholder="dbtype://user:pass@hostname:port/dbname")
        if not database_uri:
            st.info(lang_config['database_uri'])
            st.stop()
    else:
        database_dialect = "SQLite"
        database_uri = "sqlite:///Chinook.db"

    allow_query_exec = st.sidebar.checkbox(lang_config['query_exec'], value=False)

    database_config = {
        "dialect": database_dialect,
        "uri": database_uri,
        "allow_query_exec": allow_query_exec,
    }

    return model_info, model_kwargs, database_config

def print_sql_samples(documents: List[dict]) -> None:
    for doc in documents:
        try:
            page_content_dict = json.loads(doc.page_content)
            for key, value in page_content_dict.items():
                if key == 'query':
                    st.markdown(f"```\n{value}\n```")
                else:
                    st.markdown(f"{value}")
            st.markdown('<div style="margin: 5px 0;"><hr style="border: none; border-top: 1px solid #ccc; margin: 0;" /></div>', unsafe_allow_html=True)
        except json.JSONDecodeError:
            st.text("Invalid page_content format")

def initialize_os_client(enable_flag: bool, client_params: Dict, indexing_function, lang_config: Dict) -> Union[OpenSearchClient, str]:
    if enable_flag:
        client = OpenSearchClient(**client_params)
        indexing_function(client, lang_config)
    else:
        client = ""
    return client

def init_opensearch(chat_model: ChatModel) -> Tuple[Union[OpenSearchClient, str], Union[OpenSearchClient, str]]:
    with st.sidebar:
        enable_rag_query = st.sidebar.checkbox(lang_config['rag_query'], value=False)
        sql_os_client = initialize_os_client(
            enable_rag_query,
            {
                "emb": chat_model.emb,
                "index_name": 'example_queries',
                "mapping_name": 'mappings-sql',
                "vector": "input_v",
                "text": "input",
                "output": ["input", "query"]
            },
            sample_query_indexing,
            lang_config
        )

        enable_schema_desc = st.sidebar.checkbox(lang_config['schema_desc'], value=False)
        schema_os_client = initialize_os_client(
            enable_schema_desc,
            {
                "emb": chat_model.emb,
                "index_name": 'schema_descriptions',
                "mapping_name": 'mappings-schema',
                "vector": "col_desc_v",
                "text": "col_desc",
                "output": ["col", "col_desc"]
            },
            schema_desc_indexing,
            lang_config
        )

    return sql_os_client, schema_os_client


def main() -> None:
    model_info, model_kwargs, database_config = render_sidebar()
    chat_model = ChatModel(model_info, model_kwargs)
    sql_os_client, schema_os_client = init_opensearch(chat_model)

    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]
            
    display_chat_messages([])
    prompt = st.chat_input(placeholder=lang_config['example_msg'])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        db_client = DatabaseClient(chat_model.llm, database_config, sql_os_client, schema_os_client)
        samples = db_client.get_sample_queries(prompt)

        with st.chat_message("assistant"):
            with st.expander("Referenced Samples Queries (Click to expand)", expanded=False):
                print_sql_samples(samples)
            callback = StreamlitCallbackHandler(st.container())
            response = db_client.sql_executor.invoke({"question": prompt, "dialect": db_client.dialect, "samples": samples, "chat_history": st.session_state.messages}, config={"callbacks": [callback]})
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            st.write(response['output'])

if __name__ == "__main__":
    main()
