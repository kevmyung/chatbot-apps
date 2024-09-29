import streamlit as st
from py2neo import Graph
from libs.model import ChatModel
from libs.graph_workflow import build_langgraph_workflow, update_global_object, run_workflow
from urllib.parse import urlparse
from typing import List, Dict

st.set_page_config(page_title='Bedrock AI Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 Graph RAG Chatbot")

INIT_MESSAGE = {"role": "assistant", "content": "Hello! How can I help you today?"}

def new_chat():
    st.session_state.messages = [INIT_MESSAGE]
    st.session_state.langchain_messages = []

def initialize_session_state():
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = ChatModel()
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]
    if "workflow" not in st.session_state:
        st.session_state.workflow = None 

def update_settings():
    graph_url = st.session_state.graph_access
    parsed_url = urlparse(graph_url)

    scheme = parsed_url.scheme
    username = parsed_url.username
    password = parsed_url.password
    hostname = parsed_url.hostname
    port = parsed_url.port if parsed_url.port else 7687

    graph = Graph(f"{scheme}://{hostname}:{port}", auth=(username, password))
    boto3_client = st.session_state.chat_model.init_boto3_client(st.session_state.model_region)
    update_global_object(
        graph=graph, 
        boto3_client=boto3_client, 
        graph_url=f"{scheme}://{hostname}:{port}",
        graph_username=username, 
        graph_password=password
    )

    st.session_state.workflow = build_langgraph_workflow()

def config_sidebar():
    with st.sidebar:
        st.button("New Chat", on_click=new_chat, type="primary")
        st.text_input("Graph Endpoint", value="bolt://neo4j:password@localhost:7687/", key="graph_access")
        st.selectbox("Model Region", st.session_state.chat_model.get_region_list(), index=0, key="model_region")
        model_name1 = st.selectbox("Model1 (Core)", st.session_state.chat_model.get_model_list(), index=0)
        st.session_state.core_model = st.session_state.chat_model.get_model_id(model_name1)
        model_name2 = st.selectbox("Model2 (Support)", st.session_state.chat_model.get_model_list(), index=2)
        st.session_state.support_model = st.session_state.chat_model.get_model_id(model_name2)
        if st.button("Apply Settings"):
            update_settings()

def display_chat_messages(container):
    for message in st.session_state.messages:
        with container.chat_message(message["role"]):
            st.markdown(message["content"])

def generate_response(prompt: str, progress_container) -> str:
    with st.spinner('Thinking...'):
        answer = run_workflow(prompt, 
                            st.session_state.workflow, 
                            st.session_state.core_model,
                            st.session_state.support_model, 
                            st.session_state.model_region,
                            progress_container)
    return answer

def process_user_input(prompt: str, container):
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with container.chat_message("user"):
            st.markdown(prompt)
        
        with container.chat_message("assistant"):
            progress_expander = st.expander("View Progress", expanded=True)
            progress_container = progress_expander.container()
            response = generate_response(prompt, progress_container)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    initialize_session_state() 
    config_sidebar()

    if st.session_state.workflow is None:
        st.warning("Please configure settings and click 'Apply Settings' in the sidebar.")
        return

    chat_container = st.container()
    display_chat_messages(chat_container)

    user_prompt = st.chat_input("Input your message...")
    process_user_input(user_prompt, chat_container)

if __name__ == "__main__":
    main()