# System Architecture

## 1. Pipeline Detail

### 1.1. Ingestion Pipeline (데이터 적재)
데이터 소스에서 데이터를 가져와 접근 권한과 함께 벡터 DB에 저장하는 **독립 프로세스**입니다.
Serving API와 분리되어 별도의 컨테이너(Port 8001)로 동작하며, **asyncio 백그라운드 태스크**로 주기적으로 수집을 수행합니다.

1. **IngestionScheduler**: asyncio 백그라운드 태스크로 동작하며, **어댑터별 개별 주기** 설정을 지원합니다. 각 어댑터는 `{ADAPTER}_SYNC_INTERVAL_MINUTES` 환경 변수로 독립적인 수집 주기를 가질 수 있으며, 0이면 글로벌 설정(`INGESTION_INTERVAL_MINUTES`)을 사용합니다. Blocking I/O는 `asyncio.to_thread`로 별도 스레드에서 실행합니다.
2. **SyncStateDB**: MySQL 기반 DB로, 소스별 마지막 동기화 시점(Timestamp)을 저장하여 증분 수집(Incremental Ingestion)을 지원합니다. 외부 MySQL 서비스를 사용하여 데이터 일관성을 보장합니다.
3. **Adapter**: 각 소스(Slack, Jira 등)의 API 규격에 맞춰 데이터와 **접근 권한(ACL)** 정보를 가져옵니다.
4. **Chunker (Parent-Child Strategy)**:
   - **Parent Chunk**: 문맥 유지를 위해 문서를 큰 단위(2000 tokens, 설정 가능)로 분할하여 Qdrant의 **Parent Collection**에 저장합니다.
   - **Child Chunk**: 검색 정밀도를 위해 Parent를 더 작은 단위(400 tokens, 설정 가능)로 재분할하여 벡터화합니다.
5. **Embedder**: `text-embedding-3-small`을 사용하여 **Child Chunk**를 Dense Vector로 변환합니다. 배치(100건 단위) 처리를 지원합니다.
6. **SparseEncoder**: BM25 알고리즘 기반으로 **Child Chunk**의 Sparse Vector를 생성합니다. IDF 계산 및 TF 정규화를 수행합니다.
7. **VectorStore**: Qdrant에 Child Vector(Dense + Sparse Named Vectors)와 **Parent ID**, 권한 정보(ACL)를 저장합니다.

### 1.2. Serving Pipeline (검색 및 답변)
사용자 요청을 실시간으로 처리하며, 보안 필터링을 수행합니다.
1. **Auth Middleware**: 요청 헤더(`X-User-Id`, `X-User-Groups`)에서 사용자 컨텍스트를 추출합니다.
2. **Query Rewriter**:
   - LLM을 사용하여 사용자 질문을 구조화된 JSON으로 변환합니다.
   - *Decomposition*: 복잡한 질문일 경우, 여러 개의 하위 질문으로 분해하여 각각 검색을 수행합니다.
   - *Metadata Extraction*: 날짜, 작성자, 소스 등의 필터 조건을 추출합니다.
3. **Search Router**: 질문 특성에 따라 검색 전략(Hybrid/Dense/Sparse)과 소스 필터를 결정합니다.
4. **Secure Hybrid Retriever**:
   - **Permission Filter**: Qdrant 조회 시, 사용자가 접근 가능한 문서(ACL Match)만 필터링합니다.
   - **Small-to-Big Retrieval**: Child Vector로 검색 후, Parent Collection에서 원본 문맥을 조회합니다.
   - 복수 쿼리 결과를 `parent_id` 기준으로 중복 제거 후 점수순 정렬합니다.
5. **Reranker**: 1차 검색된 상위 50개 문서를 LLM 기반 관련성 점수(0~10)로 재순위화하여 Top-5를 추출합니다.
6. **Generator**: Top-5 문서와 질문을 System Prompt에 넣어 최종 답변을 생성합니다. Context 내 정보만 사용하도록 제약하며, 출처 링크를 포함합니다.

## 2. Component Design

### Query Rewriter Logic
단순 검색 실패를 방지하기 위해 LLM에게 다음과 같은 역할을 부여합니다.
- **Expansion**: 도메인 특화 용어 확장 (e.g., "HR" -> "인사팀", "Human Resources")
- **Decomposition**: 복합적인 질문을 단일 의도의 하위 질문들로 분해.
- **Metadata Extraction**: 질문에서 날짜, 작성자, 특정 채널명을 추출하여 DB 필터링 조건으로 변환.
- **Output Format**: `refined_query`, `sub_queries`, `metadata_filters`, `keywords`를 포함하는 JSON 객체.

### Advanced Indexing Strategy: Parent Document Retrieval
단순히 문서를 작게 자르면 검색은 잘 되지만 문맥이 잘리고, 크게 자르면 검색 정확도가 떨어지는 트레이드오프를 해결합니다.
1. 문서를 **Parent Chunk** (Large, 기본 2000 tokens)로 자르고 Qdrant Parent Collection에 저장합니다.
2. Parent를 다시 **Child Chunk** (Small, 기본 400 tokens)로 잘라 Dense + Sparse 벡터 인덱싱합니다.
3. 검색 시에는 **Child Vector**로 유사도를 찾고, 실제 답변 생성 시에는 매핑된 **Parent Chunk**를 LLM에게 제공하여 문맥 이해도를 극대화합니다.

### Ingestion Scheduler (Per-Adapter Intervals)
```
Ingestion Worker Start (lifespan)
  └── IngestionScheduler.start()
        └── asyncio.create_task(_loop)
              ├── [10초 대기 후 최초 실행]
              └── [매 1분마다 각 어댑터별 실행 시점 확인]
                    ├── _should_run(adapter) → 어댑터별 interval 체크
                    │     ├── SLACK_SYNC_INTERVAL_MINUTES > 0 → 개별 주기
                    │     └── 0이면 INGESTION_INTERVAL_MINUTES 사용
                    └── [실행 대상 어댑터에 대해]:
                          ├── SyncStateDB.get_last_sync(source)
                          ├── Adapter.fetch_updates(last_sync_time)  # asyncio.to_thread
                          ├── Processor.process(documents)
                          ├── Embedder.generate_embeddings(children)
                          ├── SparseEncoder.fit + encode_batch
                          ├── VectorStore.upsert_parents / upsert_children
                          └── SyncStateDB.update_last_sync(source)
```

### Vector DB Schema (Qdrant)
Child Chunk 저장을 위한 스키마입니다.
```json
{
  "id": "child_uuid",
  "vectors": {
      "dense": [0.12, 0.54, ...],
      "sparse": {
          "indices": [34, 560, ...],
          "values": [0.8, 0.5, ...]
      }
  },
  "payload": {
    "parent_id": "parent_uuid",
    "source": "jira",
    "created_at": "2024-01-05T10:00:00Z",
    "author": "dev_user",
    "url": "https://jira.company.com/browse/PROJ-123",
    "content_text": "Small child text content...",
    "permissions": {
        "users": ["alice", "bob"],
        "groups": ["engineering", "jira-users"],
        "level": "public"
    }
  }
}
```
