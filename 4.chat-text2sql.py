import streamlit as st
import os
from typing import Dict, Tuple, List, Union
from libs.db_utils import DB_Tool_Client, database_setting
from libs.config import load_model_config, load_language_config
from libs.opensearch import init_opensearch
from libs.common_utils import handle_language_change, parse_conversation_history, display_chat_messages, update_tokens_and_costs, calculate_and_display_costs, calculate_cost_from_tokens
from libs.insight_utils import analyze_main

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
    if "tokens" not in st.session_state:
        st.session_state.tokens = {'total_input_tokens': 0, 'total_output_tokens': 0, 'total_tokens': 0, 'delta_input_tokens': 0, 'delta_output_tokens': 0, 'delta_total_tokens': 0}
    if "page" not in st.session_state:
        st.session_state.page = "main"


def render_sidebar() -> Tuple[Dict, Dict, Dict]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("New Chat", on_click=new_chat, type="primary")
    with col2:
        if st.button("Analyze", type="primary"):
            st.session_state.page = "analyze"
            st.rerun()

    st.sidebar.selectbox(
        'Language 🌎',
        ['Korean', 'English'],
        key='language_select',
        on_change=handle_language_change
    )

    model_config = load_model_config()
    model_name_select = st.sidebar.selectbox(
        st.session_state.lang_config['model_selection'],
        list(model_config.keys()),
        key='model_name',
    )
    model_info = model_config[model_name_select]

    model_info["region_name"] = st.sidebar.selectbox(st.session_state.lang_config['region'], BEDROCK_REGIONS, key='bedrock_region')

    model_kwargs = {
        "temperature": 0.0,
        "top_p": 0.1,
        "top_k": 200,
        "max_tokens": 10240
    }

    database_config = database_setting(st.session_state.lang_config)
    return model_info, model_kwargs, database_config


def main():
    initialize_session_state()
    model_info, model_kwargs, database_config = render_sidebar()
    sql_os_client, schema_os_client = init_opensearch(st.session_state.lang_config)

    display_chat_messages(st.session_state.messages)
    prompt = st.chat_input(placeholder=st.session_state.lang_config['example_msg'])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        assistant_placeholder = st.empty()
        with assistant_placeholder.container():
            with st.chat_message("assistant"):
                history = parse_conversation_history(st.session_state.messages[1:][-3:])
                db_client = DB_Tool_Client(model_info, database_config, st.session_state.language_select, sql_os_client, schema_os_client, prompt, history)
                with st.expander("Scratchpad (Click to expand)", expanded=True): 
                    response_placeholder = st.empty()  
                    response, tokens = db_client.invoke(response_placeholder)
                update_tokens_and_costs(tokens)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response) 

    input_cost, output_cost, total_cost = calculate_cost_from_tokens(st.session_state.tokens, model_info['model_id'])
    calculate_and_display_costs(input_cost, output_cost, total_cost)


if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    if st.session_state.page == "main":
        main()
    if st.session_state.page == "analyze":
        analyze_main()