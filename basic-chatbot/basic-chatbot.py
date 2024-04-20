from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Tuple, Union, Dict
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import BedrockChat
import streamlit as st
import time
import os
import boto3
import json
import yaml
import random

region_name='us-east-1'
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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class ChatModel:
    def __init__(self, model_name: str, model_info: Dict, model_kwargs: Dict):
        self.model_info = model_info
        self.model_id = self.model_info["model_id"]
        self.model_kwargs = model_kwargs
        self.llm = BedrockChat(model_id=self.model_id, region_name=region_name, model_kwargs=model_kwargs, streaming=True)

    def format_prompt(self, prompt: str) -> Union[str, List[Dict]]:
        model_info = self.model_info
        if model_info.get("input_format") == "text":
            return prompt
        elif model_info.get("input_format") == "list_of_dicts":
            prompt_text = {"type": "text", "text": prompt}
            return [prompt_text]
        else:
            raise ValueError(f"Unsupported input format for model: {self.model_id}")


def load_model_config():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(file_dir, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def render_sidebar() -> Tuple[Dict, Dict]:
    with st.sidebar:
        model_config = load_model_config()
        model_name_select = st.selectbox(
            'Chat Model',
            list(model_config["models"].keys()),
            key=f"{st.session_state['widget_key']}_Model_Id",
        )
        st.session_state["model_name"] = model_name_select
        model_info = model_config["models"][model_name_select]

        system_prompt_disabled = model_config.get("system_prompt_disabled", False)
        system_prompt = st.text_area(
            "System Prompt",
            value = "You're a cool assistant, love to respond with emoji.",
            key=f"{st.session_state['widget_key']}_System_Prompt",
            disabled=system_prompt_disabled
        )

        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 250,
            "max_tokens": 4096,
        }
        if not model_info.get("system_prompt_disabled", False):
            model_kwargs["system"] = system_prompt

    return model_info, model_kwargs


def display_chat_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                display_user_message(message["content"])

            if message["role"] == "assistant":
                display_assistant_message(message["content"])   


def display_user_message(message_content: Union[str, List[dict]]) -> None:   
    if isinstance(message_content, str):
        message_text = message_content
    elif isinstance(message_content, dict):
        message_text = message_content["input"][0]["content"][0]["text"]
    else:
        message_text = message_content[0]["text"]

    message_content_markdown = message_text.split('</context>\n\n', 1)[-1]
    st.markdown(message_content_markdown)


def display_assistant_message(message_content: Union[str, dict]) -> None:
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])


def langchain_messages_format(messages: List[Union["AIMessage", "HumanMessage"]]) -> List[Union["AIMessage", "HumanMessage"]]:
    for i, message in enumerate(messages):
        if isinstance(message.content, list):
            if "role" in message.content[0]:
                if message.type == "ai":
                    message = AIMessage(message.content[0]["content"])
                if message.type == "human":
                    message = HumanMessage(message.content[0]["content"])
                messages[i] = message
    return messages


def generate_response(conversation: ConversationChain, input: Union[str, List[dict]]) -> str:
    return conversation.invoke(
        {"input": input}, {"callbacks": [StreamHandler(st.empty())]}
    )


def main() -> None:
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    model_info, model_kwargs = render_sidebar()

    chat_model = ChatModel(st.session_state["model_name"], model_info, model_kwargs)
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

    display_chat_messages()
    prompt = st.chat_input()

    if prompt:
        context_text = ""
        prompt_new = f"Here is some context for you: \n<context>\n{context_text}</context>\n\n{prompt}"
    
        formatted_prompt = chat_model.format_prompt(prompt_new)
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
