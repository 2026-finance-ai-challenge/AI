---
name: k-market-delivery
description: K-Market-Navigator AI의 RAG 구현, 평가, 문서화와 Git 전달 작업에 사용한다. dev 기반 브랜치 흐름을 적용하고 사용자 승인 전에는 커밋하지 않는다.
---

# K-Market 개발·전달 절차

1. `AGENTS.md`, `docs/PRODUCT_SCOPE.md`, `docs/GIT_WORKFLOW.md`를 읽는다.
2. Git 상태를 확인하고 최신 `dev` 기반 작업 브랜치를 사용한다.
3. 요청 범위만 구현하고 테스트·타입 검사·RAG 평가를 실행한다.
4. 변경 내역, 검증 결과, 남은 위험을 사용자에게 제시한다.
5. 사용자 승인 전까지 커밋·푸시·PR·병합을 보류한다.
6. 승인 후 한글 명사형 제목으로 커밋한다. `dev` PR 병합을 마치면 작업 브랜치를 삭제한다.
