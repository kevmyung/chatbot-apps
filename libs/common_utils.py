import streamlit as st
import re
import os
import boto3

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