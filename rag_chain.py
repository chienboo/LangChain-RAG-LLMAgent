# rag_chain.py
from llm_factory import create_llm, create_ollama_llm, create_llm_ollama_openai
from document_processor import process_document
from vector_store import create_retriever, format_docs
from prompt_template import create_rag_prompt

from langchain_core.runnables import RunnableWithMessageHistory, RunnableParallel
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from operator import itemgetter

# Step 1.  Loading LLM
# llm = create_llm()
llm = create_llm_ollama_openai()
# llm = create_ollama_llm()

# Step 2 & 3.  Loading and Splitting Documents
# 使用 document_processor 模块加载并分割文档
# 支持多种格式：txt、pdf、markdown、docx
# 默认配置从 config.py 中读取（docs/ifpc.txt, chunk_size=500, chunk_overlap=100）
splits = process_document()

# Step 4. Creating Vector Store and Retriever
# 使用 vector_store 模块创建向量库和检索器
# 自动创建嵌入模型、向量化文档并生成检索器
# 默认配置从 config.py 中读取（嵌入模型、向量库类型）
retriever = create_retriever(splits)

# Step 5. Creating RAG Prompt Template
# 使用 prompt_template 模块创建 Prompt 模板
# 包含系统消息、历史消息占位符和用户输入模板
# 默认配置从 config.py 中读取（系统提示词、用户模板）
prompt = create_rag_prompt()


# 生成 RAG chain
# 将检索到的文档内容（context）与用户输入（input）一起传递

rag_chain = (
    RunnableParallel({
        "context": itemgetter("input") | retriever | format_docs  , # 提取检索到的文档内容并打印
        "input": RunnablePassthrough(), # 用户输入
        "history": itemgetter("history") # 历史消息
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 添加调试函数：检查检索是否正常工作
def debug_retrieval(query: str):
    """调试函数：检查检索是否正常工作"""
    docs = retriever.get_relevant_documents(query)
    print(f"[DEBUG RAG] 查询: {query}")
    print(f"[DEBUG RAG] 检索到 {len(docs)} 个文档块")
    if docs:
        print(f"[DEBUG RAG] 第一个文档块预览: {docs[0].page_content[:100]}...")
    return docs


# Step 6.  Memory 
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """获取或创建会话历史记录
    
    这个函数会被 RunnableWithMessageHistory 自动调用，每次 invoke 时都会调用。
    如果 session_id 已存在，会复用已有的历史记录；如果不存在，会创建新的。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        print(f"[DEBUG] 创建新会话: {session_id[:8]}... (当前总会话数: {len(store)}, store keys: {list(store.keys())[:3]})")
    else:
        # 获取已有会话的历史记录数量，用于调试
        msg_count = len(store[session_id].messages)
        print(f"[DEBUG] 复用会话: {session_id[:8]}... (已有 {msg_count} 条历史消息, 总会话数: {len(store)})")
    return store[session_id]

def clear_session(session_id: str):
    """清空指定会话的历史记录"""
    if session_id in store:
        del store[session_id]
        print(f"[DEBUG] 已清空会话: {session_id[:8]}... (剩余会话数: {len(store)})")
    else:
        print(f"[DEBUG] 尝试清空不存在的会话: {session_id[:8]}...")

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

