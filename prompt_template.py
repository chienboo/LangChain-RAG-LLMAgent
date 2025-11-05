# prompt_template.py
"""
Prompt 模板模块
提供 RAG 系统使用的 Prompt 模板创建功能
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import PROMPT_CONFIG


def create_rag_prompt(system_message=None, user_template=None):
    """
    创建 RAG 系统的 Prompt 模板
    
    Prompt 结构：
    1. system: 系统角色定义和指令
    2. history: 历史对话消息（通过 MessagesPlaceholder 动态插入）
    3. human: 用户输入，包含检索到的文档上下文和用户问题
    
    参数:
        system_message (str, optional): 系统提示词，定义助手的行为和角色。
                                        如果为 None，使用 config.py 中的默认值
        user_template (str, optional): 用户消息模板，必须包含 {context} 和 {input} 占位符。
                                      如果为 None，使用 config.py 中的默认值
    
    返回:
        ChatPromptTemplate: Prompt 模板对象，用于构建完整的对话提示
    
    模板占位符说明:
        - {context}: 检索到的文档内容，由 format_docs() 函数格式化
        - {input}: 用户的问题或输入
        - history: 历史对话消息，通过 MessagesPlaceholder 自动管理
    
    示例:
        >>> prompt = create_rag_prompt()
        >>> prompt = create_rag_prompt(system_message="你是一个专业的法律顾问")
    """
    # 使用默认配置（如果未提供参数）
    if system_message is None:
        system_message = PROMPT_CONFIG["system_message"]
    if user_template is None:
        user_template = PROMPT_CONFIG["user_template"]
    
    # 创建 Prompt 模板
    # ChatPromptTemplate 支持多轮对话结构
    prompt = ChatPromptTemplate.from_messages([
        # 系统消息：定义助手角色和行为准则
        ("system", system_message),
        
        # 历史消息占位符：自动插入历史对话内容
        # 这允许 RAG 系统利用对话上下文进行回答
        MessagesPlaceholder("history"),
        
        # 用户消息：包含检索到的文档上下文和用户问题
        # {context} 会被检索到的文档内容替换
        # {input} 会被用户的实际问题替换
        ("human", user_template)
    ])
    
    return prompt


