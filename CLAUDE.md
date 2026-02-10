# Project Rules

## 1. Documentation Sync
코드 변경 시 관련 문서를 반드시 함께 업데이트한다.
- 모듈/컴포넌트 추가·변경·삭제 시 `docs/design.md`의 Module View, Core Components, Directory Structure를 갱신한다.
- 아키텍처 수준 변경(파이프라인 흐름, 새 서비스 등) 시 `docs/architecture.md`와 `README.md`의 다이어그램을 갱신한다.
- 기능적 요구사항 변경 시 `docs/requirements.md`를 갱신한다.
- 설정 항목 추가 시 `.env.example`과 `README.md`의 Configuration 테이블을 갱신한다.

## 2. Infrastructure Sync
컴포넌트 추가 시 인프라 코드에 반드시 반영한다.
- 새 외부 서비스(DB, 캐시 등) 추가 시 `docker-compose.yml`에 서비스, volume, network, healthcheck를 추가한다.
- 새 Python 패키지 추가 시 `requirements.txt`를 갱신한다. dev 전용이면 `requirements-dev.txt`에 추가한다.
- 새 환경 변수 추가 시 `docker-compose.yml`의 environment 섹션과 `.env.example`에 반영한다.

## 3. Configurable by Default
컴포넌트나 모듈 추가 시 하드코딩 대신 환경 변수로 설정 가능하게 만든다.
- 새 설정값은 `src/common/config.py`의 `Settings` 클래스에 필드로 추가하고, 합리적인 기본값을 지정한다.
- pydantic-settings를 통해 환경 변수에서 자동 로드되도록 한다.
- 관련 `.env.example`, `docker-compose.yml`, `README.md` Configuration 테이블도 함께 갱신한다.
