import streamlit as st
import re
import os
import boto3
import json
from typing import List, Union
from libs.config import load_language_config

def handle_language_change():
    st.session_state.lang_config = load_language_config(st.session_state.language_select)
    st.session_state.messages = [{"role": "assistant", "content": st.session_state.lang_config['init_message']}]


def sanitize_filename(filename):
    filename = os.path.splitext(filename)[0]
    sanitized = re.sub(r'[^a-zA-Z0-9\s-]', '-', filename)
    sanitized = re.sub(r'[\s-]+', '-', sanitized)
    return sanitized.strip('-')


def build_valid_message_history(messages, max_length):
    history = []
    idx = len(messages) - 1

    while idx >= 0 and messages[idx]['role'] != 'user':
        idx -= 1

    while idx >= 0 and len(history) < max_length:
        if messages[idx]['role'] == 'user':
            user_msg = messages[idx]
            idx -= 1
            history.insert(0, user_msg)
            if idx >= 0 and messages[idx]['role'] == 'assistant':
                assistant_msg = messages[idx]
                idx -= 1
                history.insert(0, assistant_msg)
            else:
                break
        else:
            idx -= 1

    while history and history[0]['role'] != 'user':
        history = history[1:]

    while history and history[-1]['role'] != 'user':
        history = history[:-1]

    if not history or history[0]['role'] != 'user' or history[-1]['role'] != 'user':
        raise ValueError("Conversation must start and end with a user message.")

    return history[-max_length:]

def get_knowledge_bases():
    if "kb_list" not in st.session_state:
        bedrock_agent_client = boto3.client('bedrock-agent', region_name=st.session_state.bedrock_region)
        response = bedrock_agent_client.list_knowledge_bases(maxResults=10)
        st.session_state.kb_list = [
            {'name': kb['name'], 'knowledgeBaseId': kb['knowledgeBaseId']} 
            for kb in response['knowledgeBaseSummaries'] 
            if kb['status'] == 'ACTIVE'
        ]
    kb_names = [kb['name'] for kb in st.session_state.kb_list]
    selected_name = st.selectbox('KnowledgeBase', kb_names)
    st.session_state.kb_id = next(kb['knowledgeBaseId'] for kb in st.session_state.kb_list if kb['name'] == selected_name)


def context_retrieval_from_kb(prompt, top_k, search_type):
    if not prompt:
        return []

    with st.chat_message("user"):
        st.markdown(prompt)

    bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name=st.session_state.bedrock_region)
    response = bedrock_agent_client.retrieve(
        knowledgeBaseId=st.session_state.kb_id,
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': top_k,
                'overrideSearchType': search_type
            }
        },
        retrievalQuery={
            'text': prompt
        }
    )        
    raw_result = response.get('retrievalResults', [])
    
    search_result = []
    context = ""

    if not raw_result:
        context = "No Relevant Context"
    else:
        for idx, result in enumerate(raw_result):
            content = result.get('content', {}).get('text', 'No content available')
            score = result.get('score', 'N/A')
            source = result.get('location', {})

            search_result.append({
                "index": idx + 1,
                "content": content,
                "source": source,
                "score": score
            })

            context += f"Result {idx + 1}:\nContent: {content}\nSource: {source}\nScore: {score}\n\n"

    prompt_new = f"Here is some context for you: \n<context>\n{context}</context>\n\n{prompt}"
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_new,
        "user_prompt_only": prompt  
    })

    return search_result


def parse_stream(stream):
    for chunk in stream:
        if 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']
            if 'text' in delta:
                yield delta['text']
        elif 'messageStop' in chunk:
            return


def invoke_model(bedrock_client, model_id, message_history, model_kwargs, history_length):
    valid_history = build_valid_message_history(message_history, history_length)
    bedrock_messages = []

    for msg in valid_history:
        content = []
        if 'content' in msg:
            content.append({'text': msg['content']})

        if 'images' in msg and msg['images']:
            for file_name in msg['images']:
                for uploaded_file in st.session_state.uploaded_files:
                    if uploaded_file.name == file_name:
                        uploaded_file.seek(0)
                        file_bytes = uploaded_file.read()
                        file_type = uploaded_file.type
                        image_format = file_type.split('/')[-1]
                        content.append({
                            'image': {
                                'format': image_format,
                                'source': {'bytes': file_bytes}
                            }
                        })

        if 'documents' in msg and msg['documents']:
            for file_name in msg['documents']:
                for uploaded_file in st.session_state.uploaded_files:
                    if uploaded_file.name == file_name:
                        uploaded_file.seek(0)
                        file_bytes = uploaded_file.read()
                        file_type = uploaded_file.type
                        doc_format = file_type.split('/')[-1]
                        if doc_format == 'plain':
                            doc_format = 'txt'
                        elif doc_format == 'x-python-script':
                            doc_format = 'txt'

                        content.append({
                            'document': {
                                'format': doc_format,
                                'name': sanitize_filename(uploaded_file.name),
                                'source': {'bytes': file_bytes}
                            }
                        })

        bedrock_messages.append({
            'role': msg['role'],
            'content': content
        })

    response = bedrock_client.converse_stream(
        modelId=model_id,
        messages=bedrock_messages,
        system=[{'text': model_kwargs['system_prompt']}],
        inferenceConfig={
            'maxTokens': model_kwargs['max_tokens'],
            'temperature': model_kwargs['temperature'],
            'topP': model_kwargs['top_p']
        }
    )
    return parse_stream(response['stream'])


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


def display_search_results(search_result):
    for result in search_result:
        with st.expander(f"Result {result['index']} (Score: {result['score']})", expanded=False):
            st.write(f"**Content:**\n{result['content']}")
            st.write(f"**Source:** {result['source']}")
            st.write(f"**Score:** {result['score']}")


def display_ai_response(bedrock_client, model_id, model_kwargs, history_length, search_result=[]):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            ai_answer = ""
            message_history = st.session_state.messages.copy()
            if search_result:
                display_search_results(search_result)
            try:
                response_stream = invoke_model(bedrock_client, model_id, message_history, model_kwargs, history_length)
                for text_chunk in response_stream:
                    ai_answer += text_chunk
                    message_placeholder.markdown(ai_answer + "▌")
                message_placeholder.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            except ValueError as e:
                st.error(str(e))

def parse_conversation_history(messages):
    history = ""
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        if isinstance(content, list):
            content = ' '.join([item.get('text', '') for item in content])
        history += f"{role}: {content}\n"
    return history


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


class ToolStreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.placeholder = self.container.empty()

    def on_llm_new_token(self, token: str) -> None:
        self.text += token
        self.placeholder.markdown(self.text)

    def on_llm_new_result(self, token: str) -> None:
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


def update_tokens_and_costs(tokens):
    st.session_state.tokens['delta_input_tokens'] = tokens['total_input_tokens']
    st.session_state.tokens['delta_output_tokens'] = tokens['total_output_tokens']
    st.session_state.tokens['total_input_tokens'] += tokens['total_input_tokens']
    st.session_state.tokens['total_output_tokens'] += tokens['total_output_tokens']
    st.session_state.tokens['delta_total_tokens'] = tokens['total_tokens']
    st.session_state.tokens['total_tokens'] += tokens['total_tokens']

def calculate_cost_from_tokens(tokens, model_id):
    PRICING = {
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "input_rate": 0.003,
            "output_rate": 0.015
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "input_rate": 0.003,
            "output_rate": 0.015
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input_rate": 0.00025,
            "output_rate": 0.00125
        },
    }
    if model_id not in PRICING:
        return 0.0, 0.0, 0.0 
    
    input_cost = tokens['total_input_tokens'] / 1000 * PRICING[model_id]['input_rate']
    output_cost = tokens['total_output_tokens'] / 1000 * PRICING[model_id]['output_rate']
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost
