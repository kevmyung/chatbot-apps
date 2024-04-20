from typing import Dict, List, Union
from langchain_community.chat_models import BedrockChat

region_name = 'us-east-1'

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
