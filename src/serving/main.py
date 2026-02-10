import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.metrics import (
    GENERATION_LATENCY,
    QUERY_REWRITE_LATENCY,
    RERANK_LATENCY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    RETRIEVED_DOCUMENTS,
    RETRIEVAL_LATENCY,
    SERVICE_INFO,
)
from src.ingestion.embedder import SparseEncoder
from src.ingestion.vector_store import VectorStore
from src.serving.core.security import UserContext, extract_user_context
from src.serving.generation.llm_client import LLMClient
from src.serving.query_engine.rewriter import QueryRewriter
from src.serving.query_engine.router import SearchRouter
from src.serving.retrieval.hybrid import HybridRetriever
from src.serving.retrieval.reranker import Reranker

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# --- Serving Components ---

vector_store = VectorStore()
sparse_encoder = SparseEncoder()
query_rewriter = QueryRewriter()
search_router = SearchRouter()
hybrid_retriever = HybridRetriever(vector_store, sparse_encoder)
reranker = Reranker()
llm_client = LLMClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 Vector DB 연결을 확인합니다."""
    logger.info("RAG Serving API 시작")

    # Set service info
    SERVICE_INFO.info({"service": "serving", "version": "1.0.0"})

    try:
        # VectorStore 연결 확인 (컬렉션은 Ingestion Worker에서 초기화)
        vector_store._client.get_collections()
        logger.info("Vector DB 연결 확인 완료")
    except Exception as e:
        logger.warning(f"Vector DB 연결 확인 실패 (Qdrant 연결 확인 필요): {e}")

    yield

    logger.info("RAG Serving API 종료")


app = FastAPI(
    title="RAG Search Service",
    description="RAG 기반 검색 API 서비스",
    debug=settings.debug,
    lifespan=lifespan,
)


# --- Data Models ---


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o")
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


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResponse(BaseModel):
    answer: str
    sources: List[dict]


# --- Endpoints ---


@app.get("/health")
async def health_check():
    """서비스 헬스 체크 엔드포인트."""
    return {"status": "ok", "service": "serving"}


@app.get("/metrics")
async def metrics():
    """Prometheus 메트릭 엔드포인트."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    user_ctx: UserContext = Depends(extract_user_context),
):
    """
    OpenAI Chat Completions API 호환 엔드포인트.
    RAG 파이프라인을 통해 검색 기반 답변을 생성합니다.
    """
    start_time = time.time()
    status = "success"

    try:
        last_user_message = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        if not last_user_message:
            status = "error"
            raise HTTPException(status_code=400, detail="User message is required")

        # 1. Query Rewriting
        rewrite_start = time.time()
        refined = query_rewriter.rewrite(last_user_message)
        QUERY_REWRITE_LATENCY.observe(time.time() - rewrite_start)

        # 2. Search Routing
        plan = search_router.plan(refined)

        # 3. Hybrid Retrieval (ACL 필터 포함)
        retrieval_start = time.time()
        retrieved = hybrid_retriever.retrieve(plan, user_ctx)
        RETRIEVAL_LATENCY.observe(time.time() - retrieval_start)
        RETRIEVED_DOCUMENTS.observe(len(retrieved))

        # 4. Reranking
        rerank_start = time.time()
        reranked = reranker.rerank(last_user_message, retrieved)
        RERANK_LATENCY.observe(time.time() - rerank_start)

        # 5. Answer Generation
        generation_start = time.time()
        result = llm_client.generate(last_user_message, reranked)
        GENERATION_LATENCY.observe(time.time() - generation_start)

        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(result.answer.split())

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=result.answer),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        logger.error(f"Chat completion 처리 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_COUNT.labels(
            endpoint="/v1/chat/completions", method="POST", status=status
        ).inc()
        REQUEST_LATENCY.labels(endpoint="/v1/chat/completions").observe(
            time.time() - start_time
        )


@app.post("/v1/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    user_ctx: UserContext = Depends(extract_user_context),
):
    """RAG 검색 엔드포인트. 질문에 대한 답변과 출처를 반환합니다."""
    start_time = time.time()
    status = "success"

    try:
        # 1. Query Rewriting
        rewrite_start = time.time()
        refined = query_rewriter.rewrite(request.query)
        QUERY_REWRITE_LATENCY.observe(time.time() - rewrite_start)

        # 2. Search Routing
        plan = search_router.plan(refined)

        # 3. Hybrid Retrieval
        retrieval_start = time.time()
        retrieved = hybrid_retriever.retrieve(plan, user_ctx)
        RETRIEVAL_LATENCY.observe(time.time() - retrieval_start)
        RETRIEVED_DOCUMENTS.observe(len(retrieved))

        # 4. Reranking
        rerank_start = time.time()
        reranked = reranker.rerank(request.query, retrieved)
        RERANK_LATENCY.observe(time.time() - rerank_start)

        # 5. Answer Generation
        generation_start = time.time()
        result = llm_client.generate(request.query, reranked)
        GENERATION_LATENCY.observe(time.time() - generation_start)

        return SearchResponse(answer=result.answer, sources=result.sources)

    except Exception as e:
        status = "error"
        logger.error(f"Search 처리 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_COUNT.labels(endpoint="/v1/search", method="POST", status=status).inc()
        REQUEST_LATENCY.labels(endpoint="/v1/search").observe(time.time() - start_time)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
