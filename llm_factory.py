# llm_factory.py
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG

def create_llm(config=None):
    if config is None:
        config = LLM_CONFIG
    
    return ChatOpenAI(
        model=config["model"],
        base_url=config["base_url"],
        temperature=config["temperature"],
        api_key=config["api_key"]
    )

