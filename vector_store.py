# vector_store.py
"""
向量存储模块
提供嵌入模型、向量库和检索器的创建功能
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_CONFIG, VECTOR_STORE_CONFIG


def create_retriever(splits, embedding_model_name=None, store_type=None):
    """
    创建文档检索器
    
    流程：
    1. 创建嵌入模型（Embeddings）
    2. 将文档块向量化并存储到向量库
    3. 从向量库创建检索器
    
    参数:
        splits (list): 已分割的文档块列表，来自 document_processor.process_document()
        embedding_model_name (str, optional): 嵌入模型名称。如果为 None，使用 config.py 中的默认值
        store_type (str, optional): 向量库类型。如果为 None，使用 config.py 中的默认值（当前仅支持 "faiss"）
    
    返回:
        VectorStoreRetriever: 检索器对象，用于根据查询检索相关文档
    
    示例:
        >>> splits = process_document()
        >>> retriever = create_retriever(splits)
        >>> retriever = create_retriever(splits, embedding_model_name="sentence-transformers/all-mpnet-base-v2")
    """
    # 使用默认配置（如果未提供参数）
    if embedding_model_name is None:
        embedding_model_name = EMBEDDING_CONFIG["model_name"]
    if store_type is None:
        store_type = VECTOR_STORE_CONFIG["store_type"]
    
    # 创建嵌入模型
    # HuggingFaceEmbeddings 使用预训练的句子嵌入模型，将文本转换为向量
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # 根据向量库类型创建向量库
    # 当前仅支持 FAISS（Facebook AI Similarity Search）
    if store_type == "faiss":
        # FAISS 是一个高效的相似度搜索库，适合本地使用
        vectorstore = FAISS.from_documents(splits, embeddings)
    else:
        raise ValueError(f"不支持的向量库类型: {store_type}。当前仅支持: faiss")
    
    # 从向量库创建检索器
    # 检索器可以根据查询文本自动检索最相关的文档块
    retriever = vectorstore.as_retriever()
    
    return retriever


def format_docs(docs):
    """
    格式化检索到的文档块，将其转换为字符串
    
    将多个 Document 对象的内容拼接成一个字符串，用于输入到 Prompt 中
    
    参数:
        docs (list): Document 对象列表，来自检索器的检索结果
    
    返回:
        str: 格式化后的文档内容字符串，多个文档块之间用双换行符分隔
    
    示例:
        >>> retrieved_docs = retriever.get_relevant_documents("查询文本")
        >>> formatted_text = format_docs(retrieved_docs)
    """
    # 将多个文档块的内容用双换行符连接
    # page_content 是 Document 对象中的文本内容
    return "\n\n".join(d.page_content for d in docs)

