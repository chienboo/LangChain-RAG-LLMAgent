# app.py
import gradio as gr
from rag_chain import rag_with_memory

def chat_fn(message, history):
    # Gradiohistory[(user, ai), ...]
    config = {"configurable": {"session_id": "user-001"}}
    response = rag_with_memory.invoke({"input": message}, config=config)
    # rag_chain 使用 StrOutputParser()，所以 response 已经是字符串
    return response

demo = gr.ChatInterface(
    fn=chat_fn,
    title="  (RAG + Memory)",
    description="LangChain v1.0 Demo",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()

