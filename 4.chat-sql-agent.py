import streamlit as st
import random
import os
from typing import Dict, Tuple, List, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from libs.agent import Agent
from libs.db_utils import DatabaseClient
from libs.config import load_model_config
from libs.models import ChatModel
from libs.chat_utils import StreamHandler, display_chat_messages, langchain_messages_format

region_name = 'us-east-1'
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
        model_info["region_name"] = region_name
        system_prompt_disabled = model_config.get("system_prompt_disabled", False)
        system_prompt = st.text_area(
            "시스템 프롬프트 (역할 지정) 👤",
            value="""You're a helpful assistant for answering questions.""",
            key=f"{st.session_state['widget_key']}_System_Prompt",
            disabled=system_prompt_disabled
        )

        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 200,
            "max_tokens": 4096,
        }
        if not model_info.get("system_prompt_disabled", False):
            model_kwargs["system"] = system_prompt

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

        sql_handler = st.radio("Text-to-SQL 처리방법 선택", ('Simple SQL Chain', 'SQL Chain', 'SQL Agent', 'Agent'))

        database_config = {
            "dialect": database_dialect,
            "uri": database_uri,
            "handler": sql_handler
        }

    return model_name_select, model_info, model_kwargs, database_config



def main() -> None:
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    model_name, model_info, model_kwargs, database_config = render_sidebar()

    chat_model = ChatModel(model_name, model_info, model_kwargs)
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
    prompt = st.chat_input(placeholder="2023년 매출 상위 10개 국가를 알려줘")

    prev_handler = st.session_state.get('handler', None)
    current_handler = database_config['handler']

    if not 'db_client' in st.session_state or prev_handler != current_handler:
        db_client = DatabaseClient(chat_model.llm, chat_model.emb, database_config)
        st.session_state['db_client'] = db_client
        st.session_state['handler'] = current_handler

    if not 'agent' in st.session_state:
        agent = Agent(chat_model.llm)
        st.session_state['agent'] = agent

    if prompt:
        
        db_client = st.session_state['db_client']
        agent = st.session_state['agent']
        if database_config['handler'] == 'Agent':
            final_prompt = agent.invoke(prompt)
        else:
            final_prompt = db_client.get_database_context(prompt)
        

        formatted_prompt = chat_model.format_prompt(final_prompt) 
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