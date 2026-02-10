# Requirements Specification

## 1. Functional Requirements (기능적 요구사항)

### 1.1. Query Processing (질문 처리)
- **자연어 입력**: 사용자는 정형화된 쿼리가 아닌 자연어로 질문할 수 있어야 한다.
- **Query Refinement (질문 최적화)**:
  - 사용자의 질문을 검색 엔진(Vector/Keyword)에 적합한 형태로 변환해야 한다.
  - LLM을 사용하여 동의어 확장, 모호성 제거, 하위 질문 분해(Decomposition)를 수행한다.

### 1.2. Data Ingestion (데이터 수집 및 적재)
- **멀티 소스 지원**: Slack, Jira API를 통해 데이터를 수집해야 한다.
- **최적화된 인덱싱**: 검색 정밀도와 답변 생성에 필요한 문맥(Context)을 모두 확보할 수 있는 방식으로 데이터를 인덱싱해야 한다.
- **메타데이터 관리**: 원본 데이터의 메타데이터(작성자, 날짜, 채널, 티켓 상태, URL 등)를 함께 저장하여 필터링에 사용한다.
- **주기적 자동 수집**: 백그라운드 스케줄러가 설정된 주기(기본 60분)마다 자동으로 데이터를 수집해야 한다.
- **증분 수집**: 매 수집 시 마지막 동기화 시점 이후의 변경분만 수집하여 효율성을 확보해야 한다.
- **수동 수집**: API 엔드포인트를 통해 즉시 수집을 트리거할 수 있어야 한다.

### 1.3. Retrieval & Generation (검색 및 답변)
- **문맥 중심 검색**: 단순 키워드 매칭을 넘어, 답변 생성에 필요한 충분한 원본 문맥을 함께 검색/참조할 수 있어야 한다.
- **Hybrid Search**: 정확도 향상을 위해 의미 기반 검색(Vector)과 키워드 기반 검색(BM25)을 결합하여 사용한다.
- **답변 생성**: 검색된 Context를 기반으로 OpenAI Chat Completion API를 사용하여 답변을 생성한다.
- **출처 표기**: 답변에 활용된 문서의 링크나 제목을 함께 제공한다.
- **Hallucination 방지**: 검색된 Context 내에서만 답변하도록 프롬프트를 제어한다.

### 1.4. API (인터페이스)
- **OpenAI 호환**: `/v1/chat/completions` 엔드포인트를 통해 OpenAI Chat Completions API와 호환되는 인터페이스를 제공한다.
- **검색 전용**: `/v1/search` 엔드포인트를 통해 답변과 출처 목록을 반환한다.
- **수동 수집**: `/v1/ingest` 엔드포인트를 통해 특정 소스의 즉시 수집을 트리거한다.

## 2. Non-Functional Requirements (비기능적 요구사항)

### 2.1. Extensibility (확장성)
- **Adapter Pattern**: 새로운 데이터 소스(예: Google Drive, Notion) 추가 시, 핵심 로직 변경 없이 어댑터 구현만으로 확장이 가능해야 한다.
- **Configurable**: 수집 주기, 청크 크기, 검색 Top-K 등 주요 파라미터를 환경 변수로 설정할 수 있어야 한다.

### 2.2. Accuracy (정확성)
- **Context Awareness**: 검색된 정보 조각뿐만 아니라 주변 문맥까지 고려하여 답변의 완결성을 높여야 한다.
- Hallucination(환각)을 최소화하기 위해 검색된 Context 내에서만 답변하도록 프롬프트를 제어한다.
- Reranking을 통해 검색 결과의 순위를 재조정한다.

### 2.3. Security (보안)
- 데이터 소스별 접근 권한(ACL)을 고려하여, 사용자가 권한이 있는 문서만 검색되도록 설계한다.
- 수집 시점에 ACL 정보를 함께 저장하고, 검색 시점에 필터를 강제 적용한다.

### 2.4. Operability (운영성)
- **Docker Compose**: 단일 명령어(`docker compose up`)로 전체 시스템(App + Qdrant)을 구동할 수 있어야 한다.
- **Health Check**: 서비스 상태를 확인할 수 있는 헬스체크 엔드포인트를 제공한다.
- **Graceful Shutdown**: 스케줄러 등 백그라운드 작업이 안전하게 종료되어야 한다.
