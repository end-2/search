.PHONY: setup install install-dev run run-serving run-ingestion test test-unit test-integration test-cov lint format clean docker-build docker-run compose-up compose-down compose-logs help

# Python 명령어 설정 (기본값: python. python3를 사용하려면 'make setup PYTHON=python3' 등으로 실행)
PYTHON ?= python

# 기본 타겟
.DEFAULT_GOAL := help

# 도움말
help:
	@echo "사용 가능한 명령어:"
	@echo ""
	@echo "  setup          - 가상환경 생성"
	@echo "  install        - 프로덕션 의존성 설치"
	@echo "  install-dev    - 개발 의존성 설치 (테스트, 린터 포함)"
	@echo ""
	@echo "  run-serving    - Serving API 로컬 실행 (포트 8000)"
	@echo "  run-ingestion  - Ingestion Worker 로컬 실행 (포트 8001)"
	@echo ""
	@echo "  test           - 전체 테스트 실행"
	@echo "  test-unit      - 단위 테스트만 실행"
	@echo "  test-integration - 통합 테스트만 실행"
	@echo "  test-cov       - 커버리지 포함 테스트"
	@echo ""
	@echo "  lint           - 코드 스타일 검사"
	@echo "  format         - 코드 자동 포맷팅"
	@echo ""
	@echo "  compose-up     - Docker Compose 전체 서비스 실행"
	@echo "  compose-down   - Docker Compose 서비스 중지"
	@echo "  compose-logs   - Docker Compose 로그 확인"
	@echo ""
	@echo "  clean          - 캐시 및 임시 파일 정리"

# 가상환경 및 의존성 설치
setup:
	$(PYTHON) -m venv venv
	@echo "가상환경이 생성되었습니다. 'source venv/bin/activate'로 활성화하세요."

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

# 로컬 실행
run: run-serving

run-serving:
	uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload

run-ingestion:
	uvicorn src.ingestion.main:app --host 0.0.0.0 --port 8001 --reload

# 테스트
test:
	pytest -v

test-unit:
	pytest tests/ingestion tests/serving -v

test-integration:
	pytest tests/integration -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing -v
	@echo "커버리지 리포트: htmlcov/index.html"

# 코드 품질 관리
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Docker 관련
docker-build:
	docker build -t rag-app .

docker-run:
	docker run -p 8000:8000 --env-file .env rag-app

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

compose-restart:
	docker compose restart

# 정리
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

