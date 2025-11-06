# app.py
import gradio as gr
from rag_chain import rag_with_memory, clear_session

def chat_fn(message, _history, request: gr.Request):
    # history 参数是 Gradio ChatInterface 自动传入的前端对话历史
    # 但我们使用 LangChain 的 RunnableWithMessageHistory 在后端管理历史记录
    # 所以这个参数虽然存在，但实际未使用（用 _history 也可以，但保留更安全）
    # request.session_hash 对于同一个浏览器标签页是固定的，所以 session 会被复用
    session_id = request.session_hash
    config = {"configurable": {"session_id": session_id}}
    
    # 处理清空会话命令
    if message.strip() in ["/clear", "清空", "/reset", "reset"]:
        print(f"[DEBUG] 收到清空命令，session_id: {session_id[:8]}...")
        clear_session(session_id)
        return "✅ 当前会话已清空，可以开始新的对话了。"
    
    # RunnableWithMessageHistory 会自动调用 get_session_history(session_id)
    # 获取历史记录，并将新的对话添加到历史中，下次调用时会复用
    print(f"[DEBUG] 处理用户输入: {message[:50]}...")
    response = rag_with_memory.invoke({"input": message}, config=config)
    
    # 调试：检查返回类型
    print(f"[DEBUG] 响应类型: {type(response)}")
    print(f"[DEBUG] 响应内容预览: {str(response)[:200]}...")
    
    # 如果返回的是字典，尝试提取内容
    if isinstance(response, dict):
        print("[DEBUG] ⚠️ 警告：响应是字典格式，尝试提取内容")
        # 可能是中间状态，尝试获取 'output' 或其他字段
        if 'output' in response:
            response = response['output']
        elif 'content' in response:
            response = response['content']
        else:
            # 如果都没有，转换为字符串
            response = str(response)
    
    # 确保返回字符串
    if not isinstance(response, str):
        response = str(response)
    
    print(f"[DEBUG] 最终响应长度: {len(response)} 字符")
    return response

demo = gr.ChatInterface(
    fn=chat_fn,
    title="  (RAG + Memory)",
    description="LangChain v1.0 Demo",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()

