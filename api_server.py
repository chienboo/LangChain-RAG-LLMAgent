from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from rag_chain import rag_with_memory, clear_session


API_VERSION = "v0.1.0"

app = FastAPI(
    title="RAG + Memory API",
    version=API_VERSION,
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    version: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        session_id = req.session_id or str(uuid.uuid4())
        config = {"configurable": {"session_id": session_id}}

        print(f"[DEBUG] /chat session_id={session_id[:8]}..., message preview={req.message[:50]}...")
        response = rag_with_memory.invoke({"input": req.message}, config=config)

        # 兼容字典与其他返回类型，统一转为字符串
        if isinstance(response, dict):
            response = response.get("output") or response.get("content") or str(response)
        elif not isinstance(response, str):
            response = str(response)

        return ChatResponse(answer=response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear(req: ClearRequest):
    try:
        print(f"[DEBUG] /clear session_id={req.session_id[:8]}...")
        clear_session(req.session_id)
        return {"cleared": True, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION)


