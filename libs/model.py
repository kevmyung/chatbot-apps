import boto3
from botocore.config import Config

MODEL_ID_MAPPING = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
    "Llama 3.1 405B Instruct": "meta.llama3-1-405b-instruct-v1:0",
    "Llama 3.1 70B Instruct": "meta.llama3-1-70b-instruct-v1:0",
    "Mistral Large 2": "mistral.mistral-large-2407-v1:0"
}

REGION = ["us-west-2", "us-east-1", "ap-northeast-1"]

class ChatModel:
    
    def get_region_list(self):
        return REGION
    
    def get_model_list(self):
        return list(MODEL_ID_MAPPING.keys())
    
    def get_model_id(self, key):
        return MODEL_ID_MAPPING[key]
    
    def init_boto3_client(self, region: str):
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", region_name=region, config=retry_config)
    
    
