import boto3
from botocore.config import Config


class Insight_Tools:
    def __init__(self, model_info):
        print("Frame")

class InsightTool_Client:
    def __init__(self, model_info, language, prompt):
        self.model = model_info['model_id']
        self.region = model_info['region_name']
        self.language = language
        self.top_k = 5
        self.tool_config = self.load_tool_config()
        self.client = self.init_boto3_client(self.region)
        self.tokens = {'total_input_tokens': 0, 'total_output_tokens': 0, 'total_tokens': 0}
        self.prompt = prompt
        #self.sys_prompt, self.usr_prompt = get_global_prompt(language, history, prompt)

    def init_boto3_client(self, region: str):
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", region_name=region, config=retry_config)


    