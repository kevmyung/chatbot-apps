import re
from typing import List, Union
import json
from PIL import Image, UnidentifiedImageError
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class ToolStreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.placeholder = self.container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text)
        
    def on_llm_new_result(self, token: str, **kwargs) -> None:
        try:
            parsed_token = json.loads(token)
            formatted_token = json.dumps(parsed_token, indent=2, ensure_ascii=False)
            self.text += "\n\n```json\n" + formatted_token + "\n```\n\n"
        except json.JSONDecodeError:
            if token.strip().upper().startswith("SELECT"):
                self.text += "\n\n```sql\n" + token + "\n```\n\n"
            elif token.strip().upper().startswith("COUNTRY,TOTALREVENUE"):
                self.text += "\n\n```\n" + token + "\n```\n\n"
            else:
                self.text += "\n\n" + token + "\n\n"
        
        self.placeholder.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")  
            self.status.markdown(doc.page_content)
            
            image_info = {
                "path": self.get_image_path(doc.metadata),
                "page": doc.metadata['page'],
                "source": doc.metadata['source']
            }
            st.session_state['image_paths'].append(image_info)
            self.status.update(state="complete")

    def get_image_path(self, metadata):
        base_folder = os.path.splitext(metadata['source'])[0] + "_images"
        image_file_path = os.path.join(base_folder, f"page_{metadata['page']}.png")
        return image_file_path

def stream_converse_messages(client, model, tool_config, messages, system, callback, tokens):
    response = client.converse_stream(
        modelId=model,
        messages=messages,
        system=system,
        toolConfig=tool_config
    )
    
    stop_reason = ""
    message = {"content": []}
    text = ''
    tool_use = {}

    for chunk in response['stream']:
        if 'messageStart' in chunk:
            message['role'] = chunk['messageStart']['role']
        elif 'contentBlockStart' in chunk:
            tool = chunk['contentBlockStart']['start']['toolUse']
            tool_use['toolUseId'] = tool['toolUseId']
            tool_use['name'] = tool['name']
        elif 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']
            if 'toolUse' in delta:
                if 'input' not in tool_use:
                    tool_use['input'] = ''
                tool_use['input'] += delta['toolUse']['input']
            elif 'text' in delta:
                text += delta['text']
                callback.on_llm_new_token(delta['text'])
        elif 'contentBlockStop' in chunk:
            if 'input' in tool_use:
                tool_use['input'] = json.loads(tool_use['input'])
                message['content'].append({'toolUse': tool_use})
                tool_use = {}
            else:
                message['content'].append({'text': text})
                text = ''
        elif 'messageStop' in chunk:
            stop_reason = chunk['messageStop']['stopReason']
        elif 'metadata' in chunk:
            tokens['total_input_tokens'] += chunk['metadata']['usage']['inputTokens']
            tokens['total_output_tokens'] += chunk['metadata']['usage']['outputTokens']
    return stop_reason, message


def display_pdf_images(img):
    if os.path.exists(img['path']):
            image = Image.open(img['path'])
            with st.expander(f"Page {img['page']} of PDF {img['source']}"):
                st.image(image, caption=f"Page {img['page']} of PDF {img['source']}", width=800)
    else:
        st.error("Requested image file does not exist.")

def display_images(
    image_ids: List[str],
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> None:
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0

    for image_id in image_ids:
        for uploaded_file in uploaded_files:
            if image_id == uploaded_file.file_id:
                if uploaded_file.type.startswith('image/'):
                    img = Image.open(uploaded_file)

                    with cols[i]:
                        st.image(img, caption="", width=75)
                        i += 1

                    if i >= num_cols:
                        i = 0
                elif uploaded_file.type in ['text/plain', 'text/csv', 'text/x-python-script']:
                    if uploaded_file.type == 'text/x-python-script':
                        st.write(f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == 'application/pdf':
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")

def parse_json_format(json_string):
    json_string = re.sub(r'"""\s*(.*?)\s*"""', r'"\1"', json_string, flags=re.DOTALL)
    json_string = re.sub(r'```json|```|</?response_format>|\n\s*', ' ', json_string)
    json_string = json_string.strip()
    match = re.search(r'({.*})', json_string)
    if match:
        json_string = match.group(1)
    else:
        return "No JSON object found in the string."

    try:
        parsed_json = json.loads(json_string)
    except json.JSONDecodeError as e:
        print("Original output: ", json_string)
        return f"JSON Parsing Error: {e}"
    return parsed_json

def display_chat_messages(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if uploaded_files and "images" in message and message["images"]:
                display_images(message["images"], uploaded_files)
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


def get_prompt_with_history(prompt, history):
    new_prompt_template = "<background>{history}</background> Question: {prompt}"
    return new_prompt_template.format(history=history, prompt=prompt)

def update_tokens_and_costs(tokens):
    st.session_state.tokens['delta_input_tokens'] = tokens['total_input_tokens']
    st.session_state.tokens['delta_output_tokens'] = tokens['total_output_tokens']
    st.session_state.tokens['total_input_tokens'] += tokens['total_input_tokens']
    st.session_state.tokens['total_output_tokens'] += tokens['total_output_tokens']
    st.session_state.tokens['delta_total_tokens'] = tokens['total_tokens']
    st.session_state.tokens['total_tokens'] += tokens['total_tokens']

def init_tokens_and_costs() -> None:
    st.session_state.tokens['delta_input_tokens'] = 0
    st.session_state.tokens['delta_output_tokens'] = 0
    st.session_state.tokens['total_input_tokens'] = 0
    st.session_state.tokens['total_output_tokens'] = 0
    st.session_state.tokens['delta_total_tokens'] = 0
    st.session_state.tokens['total_tokens'] = 0

def calculate_and_display_costs(input_cost, output_cost, total_cost):
    with st.sidebar:
        st.header("Token Usage and Cost")
        st.markdown(f"**Input Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_input_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_input_tokens']})</span> (${input_cost:.2f})", unsafe_allow_html=True)
        st.markdown(f"**Output Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_output_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_output_tokens']})</span> (${output_cost:.2f})", unsafe_allow_html=True)
        st.markdown(f"**Total Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_total_tokens']})</span> (${total_cost:.2f})", unsafe_allow_html=True)
    st.sidebar.button("Init Tokens", on_click=init_tokens_and_costs, type="primary")
