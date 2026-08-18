# Git 워크플로

## 브랜치

- `main`: 배포 브랜치
- `dev`: 개발 통합 브랜치
- 작업 브랜치: 최신 `dev`에서 생성

작업 브랜치 형식:

- `feat/<kebab-case>`
- `fix/<kebab-case>`
- `refactor/<kebab-case>`
- `test/<kebab-case>`
- `docs/<kebab-case>`
- `chore/<kebab-case>`
- `hotfix/<kebab-case>`

## 작업 순서

1. `dev`에서 작업 브랜치 생성
2. 구현과 검증
3. 사용자에게 변경 내역과 검증 결과 제시
4. 사용자 승인 후 커밋과 푸시
5. `dev` 대상 PR 생성 및 스쿼시 병합
6. 작업 브랜치 삭제
7. 배포 시 `dev`에서 `main` 대상 PR 생성

## 커밋과 PR 제목

형식:

```text
<prefix>: <한글 명사형 요약>
```

접두어:

- `feat`: 기능
- `fix`: 버그
- `refactor`: 구조 개선
- `test`: 테스트
- `docs`: 문서
- `build`: 빌드·의존성
- `ci`: CI
- `perf`: 성능
- `security`: 보안
- `chore`: 기타 설정

예시:

```text
feat: 공시 문단 검색과 인용 생성 추가
fix: 다른 공시 버전 검색 오류 수정
docs: 공시 RAG 기획 범위 정리
```

커밋 하나에는 하나의 논리적 변경만 담는다. PR 본문에는 목적, 변경 사항, 검증 결과, API·DB 영향, 위험 요소와 롤백 방법을 기록한다.
