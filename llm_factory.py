# llm_factory.py
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
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

# 创建 Ollama LLM with OpenAI style
def create_llm_ollama_openai(config=None):
    if config is None:
        config = LLM_CONFIG
    
    return ChatOpenAI(
        model="qwen:1.8b",
        temperature = 0.3,
        base_url = "http://localhost:11434/v1",
        api_key= "ollama"
    )



def create_ollama_llm(model_name="qwen:1.8b", temperature=0.3):
    """创建 Ollama ChatModel，用于与 ChatPromptTemplate 配合使用"""
    return ChatOllama(
        model=model_name,
        temperature=temperature
    )