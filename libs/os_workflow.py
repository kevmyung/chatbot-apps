import boto3
from botocore.config import Config
import json
import yaml
from string import Template
from opensearchpy import OpenSearch, RequestsHttpConnection
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
topics = ['AI/ML', 'Analytics', 'Architecture', 'Cloud Operations', 'Compute', 'Serverless & Containers', 'Database', 'Developer Tools', 'Security', 'Storage', 'Migration & Modernization', 'IoT', 'Other']
audience_types = ['Developers', 'System Administrator', 'IT Administrator', 'Data Scientists', 'Security Professionals', 'Other']
session_format = ['Breakout Session', 'Chalk Talk', 'Builder session', 'Workshop', 'Lightening Talk']

def init_bedrock_client():
    retry_config = Config(
        region_name=st.session_state.bedrock_region,
        retries={"max_attempts": 10, "mode": "standard"}
    )
    return boto3.client("bedrock-runtime", region_name=st.session_state.bedrock_region, config=retry_config)

def converse_with_bedrock(model_id, sys_prompt, usr_prompt):
    boto3_client = init_bedrock_client()
    temperature = 0.5
    top_p = 0.9
    inference_config = {"temperature": temperature, "topP": top_p}
    response = boto3_client.converse(
        modelId=model_id,
        messages=usr_prompt, 
        system=sys_prompt,
        inferenceConfig=inference_config,
    )
    return response

def load_prompt(func_name, prompt_file='libs/prompts.yaml'):
    with open(prompt_file, 'r') as file:
        prompts = yaml.safe_load(file)
    
    if func_name not in prompts:
        raise KeyError(f"Function '{func_name}' not found in the prompt file.")
    
    func_prompts = prompts[func_name]
        
    if 'sys_template' in func_prompts and 'user_template' in func_prompts:
        return func_prompts['sys_template'], func_prompts['user_template']
    elif 'system_prompt' in func_prompts and 'user_prompt' in func_prompts:
        return func_prompts['system_prompt'], func_prompts['user_prompt']
    else:
        raise KeyError(f"'sys_template' or 'system_prompt' not found for '{func_name}'")

def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

def create_prompt_without_parameter(sys_template, user_template):
    sys_prompt = [{"text": sys_template}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
    return sys_prompt, usr_prompt

def init_opensearch_client():
    host = os.getenv('OPENSEARCH_HOST')
    user = os.getenv('OPENSEARCH_USER')
    password = os.getenv('OPENSEARCH_PASSWORD')
    region = os.getenv('OPENSEARCH_REGION')

    os_client = OpenSearch(
        hosts = [{'host': host.replace("https://", ""), 'port': 443}],
        http_auth = (user, password),
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    return os_client

def search_tool_selection(question):
    sys_template, user_template = load_prompt('search_tool_selection')
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
    sys_prompt, usr_prompt = create_prompt(sys_template, user_template, question=question)
    response = converse_with_bedrock(model_id, sys_prompt, usr_prompt)
    return response['output']['message']['content'][0]['text']

def search_by_text(topics, audience_types, session_format, question):
    sys_template, user_template = load_prompt('search_by_text')
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
    #model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    #model_id="anthropic.claude-3-haiku-20240307-v1:0"
    sys_prompt, user_prompt = create_prompt(sys_template, user_template, question=question, topics=topics, audience_types=audience_types, session_format=session_format)
    response = converse_with_bedrock(model_id, sys_prompt, user_prompt)
    return response['output']['message']['content'][0]['text'], sys_prompt, user_prompt

def execute_query(os_client, index_name, search_query):
    try:
        search_result = os_client.search(
            index=index_name,
            body=search_query
        )
        return search_result, None
    except Exception as e:
        return None, str(e)

def search_by_text_as_fallback(previous_sys_prompt, previous_user_prompt, failed_query, error_message):
    sys_prompt, user_prompt = load_prompt('search_by_text_as_fallback')
    
    fallback_sys_prompt = sys_prompt.format(
        previous_sys_prompt=previous_sys_prompt,
        previous_user_prompt=previous_user_prompt,
        failed_query=failed_query,
        error_message=error_message
    )

    fallback_user_prompt = user_prompt

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    sys_prompt, usr_prompt = create_prompt_without_parameter(fallback_sys_prompt, fallback_user_prompt)
    response = converse_with_bedrock(model_id, sys_prompt, usr_prompt)

    return response['output']['message']['content'][0]['text']

def generate_answer_with_text_search(search_result, search_query, question):
    sys_prompt, user_prompt = load_prompt('generate_answer_with_text_search')
    
    search_query_str = json.dumps(search_query, indent=2)
    search_result_str = json.dumps(search_result, indent=2)

    user_prompt = user_prompt.format(
        question=question,
        search_query=search_query_str,
        search_result=search_result_str
    )

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    sys_prompt, user_prompt = create_prompt_without_parameter(sys_prompt, user_prompt)
    response = converse_with_bedrock(model_id, sys_prompt, user_prompt)

    return response['output']['message']['content'][0]['text']


def format_search_results(search_result, max_results=10):
    hits = search_result.get('hits', {}).get('hits', [])

    if hits:
        total_hits = search_result.get('hits', {}).get('total', {}).get('value', 0)
        formatted_results = []

        for hit in hits[:max_results]:
            source = hit.get('_source', {})
            code = source.get('code', 'N/A')
            title = source.get('title', 'N/A')
            synopsis = source.get('synopsis', 'N/A')
            topics = ', '.join(source.get('topics', ['N/A']))
            aws_services = ', '.join(source.get('aws_services', ['N/A']))
            target_audience = ', '.join(source.get('target_audience', ['N/A']))
            session_format = source.get('session_format', 'N/A')

            formatted_result = f"""
Code: {code}
Title: {title}
Synopsis: {synopsis}
Topics: {topics}
AWS Services: {aws_services}
Target Audience: {target_audience}
Session Format: {session_format}
"""
            formatted_results.append(formatted_result.strip())

        result_str = "\n\n".join(formatted_results)

        if total_hits > max_results:
            result_str += f"\n\n... (Showing {max_results} out of {total_hits} results)"

        return result_str
    else:
        return json.dumps(search_result, indent=2)

def vector_search(os_client, index_name, field, vector, weight, max_results):
    query = {
        "size": max_results,
        "_source": ["code", "title", "synopsis"],
        "query": {
            "knn": {
                field: {
                    "vector": vector,
                    "k": max_results
                }
            }
        }
    }
    try:
        response = os_client.search(index=index_name, body=query)
        return [(hit['_source'], hit['_score'] * weight) for hit in response['hits']['hits']]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_weighted_results(title_results, synopsis_results, max_results):
    combined_results = title_results + synopsis_results

    unique_results = {}
    for result, score in combined_results:
        code = result['code']  
        if code in unique_results:
            unique_results[code] = (result, max(score, unique_results[code][1]))
        else:
            unique_results[code] = (result, score)

    sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:max_results]


def search_by_similarity(os_client, index_name, question, max_results=10):
    boto3_client = init_bedrock_client()
    model_id = "amazon.titan-embed-text-v2:0"
    question_response = boto3_client.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": question})
    )
    question_embedding = json.loads(question_response['body'].read())['embedding']

    title_results = vector_search(os_client, index_name, "title_embedding", question_embedding, 0.4, max_results)
    synopsis_results = vector_search(os_client, index_name, "synopsis_embedding", question_embedding, 0.6, max_results)

    weighted_results = get_weighted_results(title_results, synopsis_results, max_results)
    return weighted_results

def generate_answer_with_similarity_search(search_result, question):
    sys_prompt, user_prompt = load_prompt('generate_answer_with_similarity_search')

    search_result_str = json.dumps(search_result, indent=2)

    user_prompt = user_prompt.format(
        question=question,
        search_result=search_result_str
    )

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    sys_prompt, user_prompt = create_prompt_without_parameter(sys_prompt, user_prompt)
    response = converse_with_bedrock(model_id, sys_prompt, user_prompt)

    return response['output']['message']['content'][0]['text']

def augment_query_with_llm(question):
    sys_prompt, user_prompt = load_prompt('augment_query_with_llm')
    user_prompt = user_prompt.format(question=question)

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    sys_prompt, user_prompt = create_prompt_without_parameter(sys_prompt, user_prompt)
    response = converse_with_bedrock(model_id, sys_prompt, user_prompt)

    return response['output']['message']['content'][0]['text']

def generate_answer_with_similarity_search_as_fallback(search_result, question, augmented_question):
    sys_prompt, user_prompt = load_prompt('generate_answer_with_similarity_search_as_fallback')

    search_result_str = json.dumps(search_result, indent=2)
    user_prompt = user_prompt.format(
        question=question,
        search_result=search_result_str
    )

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    sys_prompt, user_prompt = create_prompt_without_parameter(sys_prompt, user_prompt)
    response = converse_with_bedrock(model_id, sys_prompt, user_prompt)

    return response['output']['message']['content'][0]['text']

def execute_workflow(question, progress_container):
    os_client = init_opensearch_client()
    index_name = os.getenv('OPENSEARCH_INDEX')
    max_results = 10

    progress_container.info("🔍 Starting our search...")
    response = search_tool_selection(question)
    if response == "search_by_text":
        progress_container.info("⚙️ Crafting a smart query for you...")
        search_query, previous_sys_prompt, previous_user_prompt = search_by_text(topics, audience_types, session_format, question)
        #print("search_query:", search_query)
        search_result, error = execute_query(os_client, index_name, search_query)
        #print(search_result)
        if error:
            #print(f"attempt failed. Error: {error}")
            progress_container.warning("Hmm, let's try that again...")
            search_query = search_by_text_as_fallback(previous_sys_prompt, previous_user_prompt, search_query, error)
            search_result, error = execute_query(os_client, index_name, search_query)

            if error:
                progress_container.warning("No luck there. Let's try a different approach...")
                search_result = search_by_similarity(os_client, index_name, question, max_results)
                answer = generate_answer_with_similarity_search(search_result, question)
                return answer, search_result

        if not error:
            progress_container.success("📚 Generating answer...")
            search_result_str = format_search_results(search_result)
            answer = generate_answer_with_text_search(search_result_str, search_query, question)

            return answer, "Search Query:\n" + search_query + '\n\n' + search_result_str

    else:
        progress_container.info("🔍 Digging deeper for relevant info...")
        search_result = search_by_similarity(os_client, index_name, question, max_results)
        progress_container.success("📚 Generating answer...")
        answer = generate_answer_with_similarity_search(search_result, question)

        if answer == "None":
            progress_container.warning("Let's refine our search question a bit...")
            augmented_question = augment_query_with_llm(question)
            search_result = search_by_similarity(os_client, index_name, question, max_results)
            progress_container.success("📚 Generating answer...")
            answer = generate_answer_with_similarity_search_as_fallback(search_result, question, augmented_question)
    
        return answer, search_result
