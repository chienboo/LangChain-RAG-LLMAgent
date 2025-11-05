# document_processor.py
"""
文档处理模块
提供文档加载、分割等功能，支持多种文档格式（txt、pdf、markdown、docx）
"""

import os
from langchain_community.document_loaders import (
    TextLoader,              # 文本文件加载器
    PyPDFLoader,            # PDF 文件加载器
    UnstructuredMarkdownLoader,  # Markdown 文件加载器
    Docx2txtLoader          # Word 文档加载器
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCUMENT_CONFIG, TEXT_SPLITTER_CONFIG


def load_document(file_path=None, encoding=None):
    """
    根据文件扩展名自动选择合适的加载器加载文档
    
    支持的文档格式：
    - .txt: 纯文本文件
    - .pdf: PDF 文档
    - .md: Markdown 文件
    - .docx: Word 文档
    
    参数:
        file_path (str, optional): 文档文件路径。如果为 None，使用 config.py 中的默认路径
        encoding (str, optional): 文本文件编码格式。如果为 None，使用 config.py 中的默认编码
    
    返回:
        list: Document 对象列表，每个 Document 包含 page_content 和 metadata
    
    异常:
        ValueError: 当文件格式不支持时抛出
        FileNotFoundError: 当文件不存在时抛出（由加载器抛出）
    
    示例:
        >>> docs = load_document("docs/example.txt")
        >>> docs = load_document("docs/example.pdf")
        >>> docs = load_document()  # 使用默认配置
    """
    # 使用默认配置（如果未提供参数）
    if file_path is None:
        file_path = DOCUMENT_CONFIG["file_path"]
    if encoding is None:
        encoding = DOCUMENT_CONFIG["encoding"]
    
    # 提取文件扩展名并转换为小写，用于格式识别
    file_ext = os.path.splitext(file_path)[-1].lower()
    
    # 根据文件扩展名选择对应的加载器
    if file_ext == ".txt":
        # 纯文本文件：使用 TextLoader，需要指定编码
        loader = TextLoader(file_path, encoding=encoding)
    elif file_ext == ".pdf":
        # PDF 文件：使用 PyPDFLoader（需要安装 pymupdf）
        loader = PyPDFLoader(file_path)
    elif file_ext == ".md":
        # Markdown 文件：使用 UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_ext == ".docx":
        # Word 文档：使用 Docx2txtLoader（需要安装 docx2txt）
        loader = Docx2txtLoader(file_path)
    else:
        # 不支持的文件格式，抛出异常
        raise ValueError(f"不支持的文件格式: {file_ext}。支持的格式: .txt, .pdf, .md, .docx")
    
    # 加载文档并返回 Document 对象列表
    return loader.load()


def split_documents(docs, chunk_size=None, chunk_overlap=None):
    """
    使用递归字符分割器将文档分割成较小的块
    
    分割策略：
    - 优先按段落分割
    - 其次按句子分割
    - 最后按字符分割（如果块仍然太大）
    
    参数:
        docs (list): Document 对象列表，来自 load_document() 的返回值
        chunk_size (int, optional): 每个文档块的最大字符数。如果为 None，使用 config.py 中的默认值
        chunk_overlap (int, optional): 相邻块之间的重叠字符数，用于保持上下文连续性。
                                       如果为 None，使用 config.py 中的默认值
    
    返回:
        list: 分割后的 Document 对象列表
    
    示例:
        >>> docs = load_document("docs/example.txt")
        >>> splits = split_documents(docs, chunk_size=500, chunk_overlap=100)
        >>> splits = split_documents(docs)  # 使用默认配置
    """
    # 使用默认配置（如果未提供参数）
    if chunk_size is None:
        chunk_size = TEXT_SPLITTER_CONFIG["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = TEXT_SPLITTER_CONFIG["chunk_overlap"]
    
    # 创建递归字符分割器
    # RecursiveCharacterTextSplitter 会智能地尝试不同的分隔符（段落、句子、字符）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # 每个块的最大字符数
        chunk_overlap=chunk_overlap    # 相邻块之间的重叠字符数
    )
    
    # 执行分割并返回结果
    return splitter.split_documents(docs)


def process_document(file_path=None, encoding=None, chunk_size=None, chunk_overlap=None):
    """
    一站式文档处理函数：加载文档并分割成块
    
    这是 load_document() 和 split_documents() 的组合函数，提供了便捷的接口
    
    参数:
        file_path (str, optional): 文档文件路径。如果为 None，使用 config.py 中的默认路径
        encoding (str, optional): 文本文件编码格式。如果为 None，使用 config.py 中的默认编码
        chunk_size (int, optional): 每个文档块的最大字符数。如果为 None，使用 config.py 中的默认值
        chunk_overlap (int, optional): 相邻块之间的重叠字符数。如果为 None，使用 config.py 中的默认值
    
    返回:
        list: 分割后的 Document 对象列表，可直接用于向量化
    
    示例:
        >>> splits = process_document("docs/example.txt")
        >>> splits = process_document("docs/example.pdf", chunk_size=1000)
        >>> splits = process_document()  # 使用所有默认配置
    """
    # 第一步：加载文档
    docs = load_document(file_path=file_path, encoding=encoding)
    
    # 第二步：分割文档
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 返回处理后的文档块列表
    return splits

