import time
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.config import get_settings

settings = get_settings()

app = FastAPI(title="RAG Service (OpenAI Compatible)", debug=settings.debug)


# --- Data Models (OpenAI Compatible) ---


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="gpt-3.5-turbo", description="The model to use for the completion"
    )
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# --- Endpoints ---


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI Chat Completions API와 호환되는 엔드포인트입니다.
    현재는 실제 RAG 로직 대신 에코(Echo) 응답을 반환합니다.
    """
    try:
        # TODO: 여기에 실제 RAG 검색 및 생성 로직(Retrieval & Generation)을 연결해야 합니다.
        # 예:
        # 1. query = request.messages[-1].content
        # 2. retrieved_docs = vector_store.search(query)
        # 3. prompt = build_prompt(query, retrieved_docs)
        # 4. response_text = llm.generate(prompt)

        # 임시 로직: 사용자의 마지막 메시지를 확인하고 더미 응답 생성
        last_user_message = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "Hello!",
        )

        response_content = (
            f"RAG Service Response: I received your message: '{last_user_message}'. "
            "(Logic to be implemented)"
        )

        # 토큰 계산 (임시 값)
        prompt_tokens = len(last_user_message.split())
        completion_tokens = len(response_content.split())

        response_message = Message(role="assistant", content=response_content)

        choice = ChatCompletionChoice(
            index=0, message=response_message, finish_reason="stop"
        )

        return ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
