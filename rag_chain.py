# rag_chain.py
from llm_factory import create_llm
from document_processor import process_document
from vector_store import create_retriever, format_docs
from prompt_template import create_rag_prompt

from langchain_core.runnables import RunnableWithMessageHistory, RunnableParallel
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from operator import itemgetter

# Step 1.  Loading LLM
llm = create_llm()

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
        "context": itemgetter("input") | retriever | format_docs, # 提取检索到的文档内容
        "input": RunnablePassthrough(), # 用户输入
        "history": itemgetter("history") # 历史消息
    })
    | prompt
    | llm
    | StrOutputParser()
)


# Step 6.  Memory 
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

