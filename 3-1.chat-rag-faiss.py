import os
import streamlit as st
import random
from typing import Dict, Tuple, List, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from libs.config import load_model_config
from libs.models import ChatModel
from libs.chat_utils import StreamHandler, display_chat_messages, display_pdf_images, langchain_messages_format, PrintRetrievalHandler
from libs.file_utils import faiss_preprocess_document, faiss_reset_on_click

region_name = 'us-east-1'
st.set_page_config(page_title='친절한 Bedrock 챗봇', page_icon="🤖", layout="wide")
st.title("🤖 친절한 Bedrock 챗봇")

INIT_MESSAGE = {
    "role": "assistant",
    "content": "안녕하세요! 저는 Bedrock AI 챗봇입니다. 무엇을 도와드릴까요?",
}

CLAUDE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="input"),
])

st.session_state['show_image'] = True

def generate_response(conversation: ConversationChain, input: Union[str, List[dict]]) -> str:
    return conversation.invoke({"input": input}, {"callbacks": [StreamHandler(st.empty())]})

def render_sidebar() -> Tuple[Dict, Dict, List[st.runtime.uploaded_file_manager.UploadedFile]]:
    with st.sidebar:
        model_config = load_model_config()
        model_name_select = st.selectbox(
            '채팅 모델 💬',
            list(model_config["models"].keys()),
            key=f"{st.session_state['widget_key']}_Model_Id",
        )
        st.session_state["model_name"] = model_name_select
        model_info = model_config["models"][model_name_select]
        model_info["region_name"] = region_name
        system_prompt_disabled = model_config.get("system_prompt_disabled", False)
        system_prompt = st.text_area(
            "시스템 프롬프트 (역할 지정) 👤",
            value="You're a cool assistant, love to respond with emoji.",
            key=f"{st.session_state['widget_key']}_System_Prompt",
            disabled=system_prompt_disabled
        )

        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 200,
            "max_tokens": 4096,
        }
        if not system_prompt_disabled:
            model_kwargs["system"] = system_prompt

        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 0

        uploaded_files = st.file_uploader(
            "파일을 선택해주세요 📎",
            type=["pdf"],
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"],
        )

        st.button("지식 초기화", on_click=faiss_reset_on_click)

    return model_info, model_kwargs, uploaded_files

def main() -> None:
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    model_info, model_kwargs, uploaded_files = render_sidebar()

    chat_model = ChatModel(st.session_state["model_name"], model_info, model_kwargs)
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

    retriever = faiss_preprocess_document(uploaded_files, chat_model)

    if prompt:
        context_text = ""
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if retriever is not None:
            st.session_state['image_paths'] = []
            retrieval_handler = PrintRetrievalHandler(st.container())
            context_text = retriever.get_relevant_documents(prompt, callbacks=[retrieval_handler])

        prompt_new = f"Here's some context for you. However, do not mention this context unless it is directly relevant to the user's question. It is essential to deliver an answer that precisely addresses the user's needs \n<context>\n{context_text}</context>\n\n{prompt}\n\n"

        formatted_prompt = chat_model.format_prompt(prompt_new)
        st.session_state.messages.append({"role": "user", "content": formatted_prompt})
        
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
            if 'image_paths' in st.session_state:
                for img in st.session_state['image_paths']:
                    display_pdf_images(img)

if __name__ == "__main__":
    main()
