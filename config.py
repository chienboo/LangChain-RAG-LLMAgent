# config.py
# LLM 配置
LLM_CONFIG = {
    "model": "qwen-plus",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 0.3,
    "api_key": "sk-feb37412e7584076accb6765a04c4167"
}

# 文档加载配置
DOCUMENT_CONFIG = {
    "file_path": "docs/ifpc.txt",  # 默认文档路径
    "encoding": "utf-8"            # 文本文件编码格式
}

# 文本分割配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,      # 文档块大小（字符数）
    "chunk_overlap": 100    # 文档块之间的重叠大小（字符数）
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型名称
}

# 向量库配置
VECTOR_STORE_CONFIG = {
    "store_type": "faiss"  # 向量库类型（当前支持：faiss）
}

# Prompt 模板配置
PROMPT_CONFIG = {
    "system_message": "你是一个中文助手，需要根据用户的问题，从以下文本中回答问题。如果文本中没有答案，请如实回答不知道。",
    "user_template": "以下是与问题相关的参考资料：\n{context}\n\n问题：{input}"
}

