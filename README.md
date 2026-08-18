# K-Market-Navigator AI

한국 상장기업 공시를 구조화하고 검색·재정렬하여 근거가 표시된 답변을 생성하는 FastAPI 기반 공시 RAG 서비스다.

현재 구현 범위는 공시 파싱·청킹·임베딩·검색과 공시 전용 질의응답으로 제한한다.

## 프로젝트 문서

- [제품 범위](docs/PRODUCT_SCOPE.md)
- [Git 및 전달 워크플로](docs/GIT_WORKFLOW.md)
- [Codex 저장소 지침](AGENTS.md)
- [Codex 작업 스킬](.agents/skills/k-market-delivery/SKILL.md)

## 개발 환경

- Python 3.14
- FastAPI 0.139
- uv

## 실행

```shell
cp .env.example .env
uv sync --locked
uv run uvicorn k_market_ai.main:app --host 127.0.0.1 --port 8000
uv run k-market-rag-worker
```

`.env`는 로컬에서만 사용하며 Git에 포함하지 않는다. 배포 환경의 설정은 실행 시점에 외부에서 주입한다.

API 서버와 공시 색인 워커는 별도 프로세스로 실행한다. 상태 확인은 `GET /health`를 사용한다. 운영 환경에서는 API 문서가 비활성화된다.

## 검증

```shell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
