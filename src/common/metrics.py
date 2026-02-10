"""
Prometheus metrics definitions for RAG service.
Serving과 Ingestion 프로세스에서 공통으로 사용되는 메트릭을 정의합니다.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# --- Common Metrics ---

SERVICE_INFO = Info("rag_service", "RAG service information")

# --- Serving Metrics ---

# Request metrics
REQUEST_COUNT = Counter(
    "rag_serving_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_serving_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# RAG pipeline metrics
QUERY_REWRITE_LATENCY = Histogram(
    "rag_query_rewrite_duration_seconds",
    "Query rewrite latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RERANK_LATENCY = Histogram(
    "rag_rerank_duration_seconds",
    "Reranking latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "Answer generation latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

RETRIEVED_DOCUMENTS = Histogram(
    "rag_retrieved_documents_count",
    "Number of documents retrieved per query",
    buckets=[1, 5, 10, 25, 50, 100],
)

# --- Ingestion Metrics ---

INGESTION_RUNS = Counter(
    "rag_ingestion_runs_total",
    "Total number of ingestion runs",
    ["source", "status"],
)

INGESTION_DURATION = Histogram(
    "rag_ingestion_duration_seconds",
    "Ingestion duration in seconds",
    ["source"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total number of documents ingested",
    ["source"],
)

CHUNKS_CREATED = Counter(
    "rag_chunks_created_total",
    "Total number of chunks created",
    ["source", "type"],
)

EMBEDDING_LATENCY = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding generation latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

SCHEDULER_STATUS = Gauge(
    "rag_scheduler_running",
    "Whether the ingestion scheduler is running",
)

LAST_INGESTION_TIMESTAMP = Gauge(
    "rag_last_ingestion_timestamp_seconds",
    "Unix timestamp of last successful ingestion",
    ["source"],
)

# --- Vector DB Metrics ---

QDRANT_OPERATION_LATENCY = Histogram(
    "rag_qdrant_operation_duration_seconds",
    "Qdrant operation latency in seconds",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

QDRANT_OPERATION_COUNT = Counter(
    "rag_qdrant_operations_total",
    "Total number of Qdrant operations",
    ["operation", "status"],
)
