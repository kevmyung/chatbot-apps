from typing import Dict, List, Union
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings

class ChatModel:
    def __init__(self, model_name: str, model_info: Dict, model_kwargs: Dict):
        self.model_info = model_info
        self.model_id = self.model_info["model_id"]
        self.model_kwargs = model_kwargs
        self.llm = ChatBedrock(model_id=self.model_id, region_name=self.model_info['region_name'], model_kwargs=model_kwargs, streaming=True)
        self.emb = BedrockEmbeddings(model_id="amazon.titan-embed-g1-text-02", region_name=self.model_info['region_name'])
        
    def format_prompt(self, prompt: str) -> Union[str, List[Dict]]:
        model_info = self.model_info
        if model_info.get("input_format") == "text":
            return prompt
        elif model_info.get("input_format") == "list_of_dicts":
            prompt_text = {"type": "text", "text": prompt}
            return [prompt_text]
        else:
            raise ValueError(f"Unsupported input format for model: {self.model_id}")