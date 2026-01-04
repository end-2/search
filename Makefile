.PHONY: setup install run test lint clean docker-build docker-run

# Python 명령어 설정 (기본값: python. python3를 사용하려면 'make setup PYTHON=python3' 등으로 실행)
PYTHON ?= python

# 가상환경 및 의존성 설치
setup:
	$(PYTHON) -m venv venv
	@echo "가상환경이 생성되었습니다. 'source venv/bin/activate'로 활성화하세요."

install:
	$(PYTHON) -m pip install -r requirements.txt

# 로컬 실행
run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 코드 품질 관리
lint:
	flake8 src/
	black --check src/
	isort --check-only src/
	mypy src/

format:
	black src/
	isort src/

# Docker 관련
docker-build:
	docker build -t rag-app .

docker-run:
	docker run -p 8000:8000 --env-file .env rag-app

compose-up:
	docker-compose up -d --build

compose-down:
	docker-compose down

# 정리
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

