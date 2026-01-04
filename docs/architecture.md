# System Architecture

## 1. Pipeline Detail

### 1.1. Ingestion Pipeline (데이터 적재)
데이터 소스에서 데이터를 가져와 접근 권한과 함께 벡터 DB에 저장하는 비동기 프로세스입니다.
1. **Collector**: 스케줄러에 의해 주기적으로 실행됩니다.
2. **Adapter**: 각 소스(Slack, Jira 등)의 API 규격에 맞춰 데이터와 **접근 권한(ACL)** 정보를 가져옵니다.
3. **Chunker (Parent-Child Strategy)**: 
   - **Parent Chunk**: 문맥 유지를 위해 문서를 큰 단위(1000~2000 tokens)로 분할하여 Vector DB의 **Parent Collection**에 저장합니다.
   - **Child Chunk**: 검색 정밀도를 위해 Parent를 더 작은 단위(200~400 tokens)로 재분할하여 벡터화합니다.
    4. **Embedder**: `text-embedding-3-small` 등을 사용하여 **Child Chunk**를 벡터로 변환합니다.
    5. **Sparse Encoder**: 키워드 매칭을 위해 **Child Chunk** 기반의 Sparse Vector를 생성합니다.
    6. **Vector DB**: Child Vector(Dense/Sparse)와 **Parent ID**, 권한 정보(Allow List)를 저장합니다.

### 1.2. Serving Pipeline (검색 및 답변)
사용자 요청을 실시간으로 처리하며, 보안 필터링을 수행합니다.
1. **Query Refiner**: 
   - User Input: "지난주 배포 에러 뭐였지?"
   - LLM Output: `(date: 2024-01-01..2024-01-07) AND (keyword: "Deployment Error" OR "Fail")`
   - *Decomposition*: 복잡한 질문일 경우, 여러 개의 하위 질문으로 분해하여 각각 검색을 수행합니다.
2. **Secure Retriever**: 
   - **User Context Injection**: 요청한 사용자의 ID/그룹 정보를 파악합니다.
   - **Permission Filter**: Vector DB 조회 시, 사용자가 접근 가능한 문서(ACL Match)만 필터링하여 검색합니다.
   - **Hybrid Search**: 필터링된 범위 내에서 Vector + Keyword 검색 수행.
3. **Reranker (Optional)**: 검색된 상위 50개 문서를 Cross-Encoder로 정밀 재순위화 하여 Top-5 추출.
4. **Generator**: Top-5 문서와 질문을 System Prompt에 넣어 최종 답변 생성 (출처 링크 포함).

## 2. Component Design

### Query Refiner Logic
단순 검색 실패를 방지하기 위해 LLM에게 다음과 같은 역할을 부여합니다.
- **Expansion**: 도메인 특화 용어 확장 (e.g., "HR" -> "인사팀", "Human Resources")
- **Decomposition**: 복합적인 질문을 단일 의도의 하위 질문들로 분해.
- **Metadata Extraction**: 질문에서 날짜, 작성자, 특정 채널명을 추출하여 DB 필터링 조건으로 변환.

### Advanced Indexing Strategy: Parent Document Retrieval
단순히 문서를 작게 자르면 검색은 잘 되지만 문맥이 잘리고, 크게 자르면 검색 정확도가 떨어지는 트레이드오프를 해결합니다.
1. 문서를 **Parent Chunk** (Large, e.g., 2000 tokens)로 자르고 DocStore에 저장합니다.
2. Parent를 다시 **Child Chunk** (Small, e.g., 400 tokens)로 잘라 벡터 인덱싱합니다.
3. 검색 시에는 **Child Vector**로 유사도를 찾고, 실제 답변 생성 시에는 매핑된 **Parent Chunk**를 LLM에게 제공하여 문맥 이해도를 극대화합니다.

### Vector DB Schema Example (Qdrant Style)
Child Chunk 저장을 위한 스키마 예시입니다.
```json
{
  "id": "child_uuid",
  "vectors": {
      "dense": [0.12, 0.54, ...],  // Semantic (Child Text)
      "sparse": {                   // Keyword (Child Text)
          "indices": [34, 560, ...],
          "values": [0.8, 0.5, ...]
      }
  },
  "payload": {  // Metadata
    "parent_id": "parent_uuid",     // Reference to Full Context
    "source": "jira",
    "created_at": "2024-01-05T10:00:00Z",
    "author": "dev_user",
    "url": "[https://jira.company.com/browse/PROJ-123](https://jira.company.com/browse/PROJ-123)",
    "content_text": "Small child text content...",
    "permissions": {
        "users": ["alice", "bob"],
        "groups": ["engineering", "jira-users"],
        "level": "public" 
    }
  }
}
```
