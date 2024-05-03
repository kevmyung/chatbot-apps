import streamlit as st
import random
import os
from typing import Dict, Tuple, List, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from libs.db_utils import DatabaseClient
from libs.config import load_model_config
from libs.models import ChatModel
from libs.chat_utils import StreamHandler, display_chat_messages


st.session_state.region_name = 'us-east-1'
st.set_page_config(page_title='친절한 Bedrock 챗봇', page_icon="🤖", layout="wide")
st.title("🤖 친절한 Bedrock 챗봇")

INIT_MESSAGE = {
    "role": "assistant",
    "content": "안녕하세요! 저는 Bedrock AI 챗봇입니다. 무엇을 도와드릴까요?",
}

CLAUDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ]
)

def generate_response(conversation: ConversationChain, input: Union[str, List[dict]]) -> str:
    return conversation.invoke(
        {"input": input}, {"callbacks": [StreamHandler(st.empty())]}
    )

def render_sidebar() -> Tuple[str, Dict, Dict, Dict]:
    with st.sidebar:
        model_config = load_model_config()
        model_name_select = st.selectbox(
            '채팅 모델 💬',
            list(model_config["models"].keys()),
            key=f"{st.session_state['widget_key']}_Model_Id",
        )
        model_info = model_config["models"][model_name_select]
        model_info["region_name"] = st.session_state.region_name

        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 200,
            "max_tokens": 4096,
            "system": """
            You are a helpful assistant for answering questions in Korean. 
            Please provide a response in the <final_answer></final_answer) section, explaining which columns the process that led to the final answer to resolve the user's question. 
            Firstly, provide the detailed answer to the user's question with numbers, if possible. Next, provide the used SQL queries within a code block."""
         }

        database_selection = st.selectbox(
            '데이터베이스',
            ('SQLite-샘플', 'MySQL', 'PostgreSQL', 'SQLite', 'Presto', 'Oracle')
        )

        if database_selection != "SQLite-샘플":
            database_dialect = database_selection
            database_uri = st.text_input("Database URI", value="", placeholder="dbtype://user:pass@hostname:port/dbname")
        else:
            database_dialect = "SQLite"
            database_uri = "sqlite:///Chinook.db"

        database_config = {
            "dialect": database_dialect,
            "uri": database_uri
        }

    return model_name_select, model_info, model_kwargs, database_config

def main() -> None:
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    model_name, model_info, model_kwargs, database_config = render_sidebar()
    chat_model = ChatModel(model_name, model_info, model_kwargs)

    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE] 

    if not 'db_client' in st.session_state:
        db_client = DatabaseClient(chat_model.llm, chat_model.emb, database_config)
        st.session_state['db_client'] = db_client
    else:
        db_client = st.session_state['db_client']

    display_chat_messages([])  

    prompt = st.chat_input(placeholder="2023년 매출 상위 10개 국가를 알려줘")
    
    if prompt:        
        st.session_state.messages.append({"role": "user", "content": prompt})         
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            callback = StreamlitCallbackHandler(st.container())
            response = db_client.agent_executor({"question":prompt, "chat_history": st.session_state.messages}, callbacks=[callback])
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            st.write(response['output'])

if __name__ == "__main__":
    main()