import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Type

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.metrics import (
    CHUNKS_CREATED,
    DOCUMENTS_INGESTED,
    EMBEDDING_LATENCY,
    INGESTION_DURATION,
    INGESTION_RUNS,
    LAST_INGESTION_TIMESTAMP,
    SCHEDULER_STATUS,
    SERVICE_INFO,
)
from src.ingestion.adapters.base import DataSourceAdapter
from src.ingestion.adapters.jira import JiraAdapter
from src.ingestion.adapters.slack import SlackAdapter
from src.ingestion.embedder import Embedder, SparseEncoder
from src.ingestion.processor import Processor
from src.ingestion.scheduler import IngestionScheduler
from src.ingestion.vector_store import VectorStore

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# --- Ingestion Components ---

vector_store = VectorStore()
sparse_encoder = SparseEncoder()
processor = Processor()
embedder = Embedder()

# Adapter 레지스트리 (enabled 상태인 것만 포함)
ADAPTER_REGISTRY: Dict[str, Type[DataSourceAdapter]] = {}

if settings.slack_enabled and settings.slack_bot_token:
    ADAPTER_REGISTRY["slack"] = SlackAdapter
    logger.info("Slack Adapter 활성화")

if settings.jira_enabled and settings.jira_base_url:
    ADAPTER_REGISTRY["jira"] = JiraAdapter
    logger.info("Jira Adapter 활성화")

scheduler = IngestionScheduler(
    adapters=ADAPTER_REGISTRY,
    processor=processor,
    embedder=embedder,
    sparse_encoder=sparse_encoder,
    vector_store=vector_store,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 Vector DB 초기화 및 Ingestion 스케줄러를 기동합니다."""
    logger.info("RAG Ingestion Worker 시작")

    # Set service info
    SERVICE_INFO.info({"service": "ingestion", "version": "1.0.0"})

    # Vector DB 컬렉션 초기화
    try:
        vector_store.init_collections()
        logger.info("Vector DB 컬렉션 초기화 완료")
    except Exception as e:
        logger.warning(f"Vector DB 초기화 실패 (Qdrant 연결 확인 필요): {e}")

    # 스케줄러 기동
    if settings.ingestion_enabled:
        await scheduler.start()
        SCHEDULER_STATUS.set(1)
    else:
        SCHEDULER_STATUS.set(0)
        logger.info("Ingestion 스케줄러 비활성화 (INGESTION_ENABLED=false)")

    yield

    await scheduler.stop()
    SCHEDULER_STATUS.set(0)
    logger.info("RAG Ingestion Worker 종료")


app = FastAPI(
    title="RAG Ingestion Worker",
    description="데이터 수집 워커 서비스",
    debug=settings.debug,
    lifespan=lifespan,
)


# --- Data Models ---


class IngestionRequest(BaseModel):
    source: str = Field(description="데이터 소스 ('slack' | 'jira')")


class IngestionResponse(BaseModel):
    status: str
    documents_ingested: int
    parent_chunks: int
    child_chunks: int


class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    adapters: list
    global_interval_minutes: int


# --- Endpoints ---


@app.get("/health")
async def health_check():
    """서비스 헬스 체크 엔드포인트."""
    return {"status": "ok", "service": "ingestion"}


@app.get("/metrics")
async def metrics():
    """Prometheus 메트릭 엔드포인트."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/scheduler/status", response_model=SchedulerStatusResponse)
async def scheduler_status():
    """스케줄러 상태를 반환합니다."""
    return SchedulerStatusResponse(
        enabled=settings.ingestion_enabled,
        running=scheduler._running,
        adapters=list(ADAPTER_REGISTRY.keys()),
        global_interval_minutes=settings.ingestion_interval_minutes,
    )


@app.post("/v1/ingest", response_model=IngestionResponse)
async def ingest(request: IngestionRequest):
    """데이터 소스에서 문서를 수동으로 수집하여 Vector DB에 적재합니다."""
    start_time = time.time()
    status = "success"

    # 전체 어댑터 목록 (enabled 여부와 관계없이)
    all_adapters = {
        "slack": SlackAdapter,
        "jira": JiraAdapter,
    }

    if request.source not in all_adapters:
        INGESTION_RUNS.labels(source=request.source, status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 소스: {request.source}. 지원 소스: {list(all_adapters.keys())}",
        )

    try:
        # 1. 데이터 수집
        adapter = all_adapters[request.source]()
        raw_documents = adapter.fetch_updates()

        if not raw_documents:
            INGESTION_RUNS.labels(source=request.source, status="success").inc()
            INGESTION_DURATION.labels(source=request.source).observe(
                time.time() - start_time
            )
            return IngestionResponse(
                status="success",
                documents_ingested=0,
                parent_chunks=0,
                child_chunks=0,
            )

        # 2. 전처리 및 청킹
        parents, children = processor.process(raw_documents)

        if not children:
            INGESTION_RUNS.labels(source=request.source, status="success").inc()
            INGESTION_DURATION.labels(source=request.source).observe(
                time.time() - start_time
            )
            DOCUMENTS_INGESTED.labels(source=request.source).inc(len(raw_documents))
            CHUNKS_CREATED.labels(source=request.source, type="parent").inc(
                len(parents)
            )
            return IngestionResponse(
                status="success",
                documents_ingested=len(raw_documents),
                parent_chunks=len(parents),
                child_chunks=0,
            )

        # 3. Embedding 생성
        embedding_start = time.time()
        dense_vectors = embedder.generate_embeddings(children)
        EMBEDDING_LATENCY.observe(time.time() - embedding_start)

        # Sparse Vector 학습 및 생성
        sparse_encoder.fit([c.content for c in children])
        sparse_vectors = sparse_encoder.encode_batch(children)

        # 4. Vector DB 적재
        vector_store.upsert_parents(parents)
        vector_store.upsert_children(children, dense_vectors, sparse_vectors)

        # Record metrics
        DOCUMENTS_INGESTED.labels(source=request.source).inc(len(raw_documents))
        CHUNKS_CREATED.labels(source=request.source, type="parent").inc(len(parents))
        CHUNKS_CREATED.labels(source=request.source, type="child").inc(len(children))
        LAST_INGESTION_TIMESTAMP.labels(source=request.source).set(time.time())

        return IngestionResponse(
            status="success",
            documents_ingested=len(raw_documents),
            parent_chunks=len(parents),
            child_chunks=len(children),
        )

    except Exception as e:
        status = "error"
        logger.error(f"Ingestion 실패 ({request.source}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INGESTION_RUNS.labels(source=request.source, status=status).inc()
        INGESTION_DURATION.labels(source=request.source).observe(
            time.time() - start_time
        )


@app.post("/v1/ingest/all")
async def ingest_all():
    """활성화된 모든 소스에서 수동 수집을 실행합니다."""
    results = {}

    for source_name in ADAPTER_REGISTRY:
        try:
            response = await ingest(IngestionRequest(source=source_name))
            results[source_name] = {
                "status": response.status,
                "documents": response.documents_ingested,
                "parent_chunks": response.parent_chunks,
                "child_chunks": response.child_chunks,
            }
        except HTTPException as e:
            results[source_name] = {"status": "error", "detail": e.detail}
        except Exception as e:
            results[source_name] = {"status": "error", "detail": str(e)}

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ingestion.main:app",
        host=settings.host,
        port=settings.ingestion_port,
        reload=settings.debug,
    )
