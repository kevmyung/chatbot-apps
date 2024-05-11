import streamlit as st
import random
import os
from typing import Dict, Tuple, List, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
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

def new_chat() -> None:
    """
    Reset the chat session and initialize a new conversation chain.
    """
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []

def generate_response(conversation: ConversationChain, input: Union[str, List[dict]]) -> str:
    return conversation.invoke(
        {"input": input}, {"callbacks": [StreamHandler(st.empty())]}
    )

def render_sidebar() -> Tuple[str, Dict, Dict, Dict]:
    st.sidebar.button("채팅 초기화", on_click=new_chat, type="primary")
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
            Please provide a response in the <final_answer></final_answer> section.
            Firstly, exaplain the process that led to the final answer. 
            If you used the SQL tools for resolving the user's question, provide the detailed answer to the user's question with numbers and the used SQL queries within a Markdown code block."""
         }

        database_selection = st.selectbox(
            '데이터베이스',
            ('SQLite-샘플', 'MySQL', 'PostgreSQL', 'Redshift', 'SQLite', 'Presto', 'Oracle')
        )

        if database_selection != "SQLite-샘플":
            database_dialect = database_selection
            database_uri = st.text_input("Database URI", value="", placeholder="dbtype://user:pass@hostname:port/dbname")
            if not database_uri:
                st.info("데이터베이스 URI를 입력해주세요")
                st.stop()
        else:
            database_dialect = "SQLite"
            database_uri = "sqlite:///Chinook.db"

        with st.sidebar:
            add_schema_desc = st.checkbox("스키마 설명 추가", value=False)

            if add_schema_desc:
                schema_file = st.text_input("스키마 파일 경로", value="libs/default-schema.json")

                if not os.path.exists(schema_file):
                    st.error("스키마 파일 이름을 확인해주세요!")

            else:
                schema_file = ""

        with st.sidebar:
            allow_query_exec = st.checkbox("생성된 쿼리 실행 허용", value=True)

        database_config = {
            "dialect": database_dialect,
            "uri": database_uri,
            "schema_file": schema_file,
            "allow_query_exec": allow_query_exec
        }

    return model_name_select, model_info, model_kwargs, database_config

def main() -> None:
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    model_name, model_info, model_kwargs, database_config = render_sidebar()
    chat_model = ChatModel(model_name, model_info, model_kwargs)

    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE] 

    db_client = DatabaseClient(chat_model.llm, chat_model.emb, database_config)

    display_chat_messages([])  

    prompt = st.chat_input(placeholder="2023년 매출 상위 10개 국가를 알려줘")
    
    if prompt:        
        st.session_state.messages.append({"role": "user", "content": prompt})         
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            callback = StreamlitCallbackHandler(st.container())
            response = db_client.agent_executor.invoke({"question":prompt, "dialect":db_client.dialect, "chat_history": st.session_state.messages}, config={"callbacks": [callback]})
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            st.write(response['output'])

if __name__ == "__main__":
    main()