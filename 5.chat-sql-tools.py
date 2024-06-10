import streamlit as st
import json
from typing import Dict, Tuple, List, Union
from libs.db_utils import DatabaseClient_v2
from libs.config import load_model_config, load_language_config
from libs.models import ChatModel, calculate_cost_from_tokens
from libs.opensearch import init_opensearch
from libs.chat_utils import display_chat_messages, get_prompt_with_history, ToolStreamHandler

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
        "max_tokens": 20480
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

    database_config = {
        "dialect": database_dialect,
        "uri": database_uri
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

def update_tokens_and_costs(tokens):
    st.session_state.tokens['delta_input_tokens'] = tokens['total_input_tokens']
    st.session_state.tokens['delta_output_tokens'] = tokens['total_output_tokens']
    st.session_state.tokens['total_input_tokens'] += tokens['total_input_tokens']
    st.session_state.tokens['total_output_tokens'] += tokens['total_output_tokens']
    st.session_state.tokens['delta_total_tokens'] = tokens['total_tokens']
    st.session_state.tokens['total_tokens'] += tokens['total_tokens']


def calculate_and_display_costs(model_id):
    input_cost, output_cost, total_cost = calculate_cost_from_tokens(st.session_state.tokens, model_id)

    with st.sidebar:
        st.header("Token Usage and Cost")
        st.markdown(f"**Input Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_input_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_input_tokens']})</span> (${input_cost:.2f})", unsafe_allow_html=True)
        st.markdown(f"**Output Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_output_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_output_tokens']})</span> (${output_cost:.2f})", unsafe_allow_html=True)
        st.markdown(f"**Total Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_total_tokens']})</span> (${total_cost:.2f})", unsafe_allow_html=True)


def main() -> None:
    model_info, model_kwargs, database_config = render_sidebar()
    chat_model = ChatModel(model_info, model_kwargs)
    sql_os_client, schema_os_client = init_opensearch(chat_model.emb, lang_config)

    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]
    if "tokens" not in st.session_state:
        st.session_state.tokens = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'delta_input_tokens': 0,
            'delta_output_tokens': 0,
            'delta_total_tokens': 0
        }

    display_chat_messages([])
    prompt = st.chat_input(placeholder=lang_config['example_msg'])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        db_client = DatabaseClient_v2(model_info, database_config, st.session_state['language_select'], sql_os_client, schema_os_client)
        samples = db_client.get_sample_queries(prompt)
        
        assistant_placeholder = st.empty()
        with assistant_placeholder.container():
            with st.chat_message("assistant"):
                with st.expander("Referenced Sample Queries (Click to expand)", expanded=False):
                    print_sql_samples(samples)

                with st.expander("Scratchpad (Click to expand)", expanded=False):  
                    response_placeholder = st.empty() 
                    callback = ToolStreamHandler(response_placeholder)

                chat_history = st.session_state.messages[-3:]
                new_prompt = get_prompt_with_history(prompt, chat_history)
                response, tokens = db_client.invoke(new_prompt, callback)
                
                update_tokens_and_costs(tokens)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

    calculate_and_display_costs(model_info['model_id'])


if __name__ == "__main__":
    main()
