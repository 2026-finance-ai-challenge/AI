# 세무 문서 OCR·교차검증 모델

## 운영 런타임

KART의 세무 문서 검증은 하나금융 프로젝트의 `src/hanah_tax_ocr` 런타임을 기준 커밋 `ab82ccc51cb096872f9a110a85c027a4158a147f`에서 이식한 로컬 모델이다. 서비스 코드에서는 제품 이름을 KART로 통일했으며 다음 런타임을 포함한다.

- Tesseract `eng`, `kor+eng`와 PDF용 Poppler
- IRS 거주자증명서, 미국 주정부 아포스티유, 국내 제한세율 적용신청서 템플릿 분류
- 문서별 영역 OCR과 전용 파서
- 흐림, OCR 신뢰도, 서명, 인장, 체크박스 품질 검사
- 필수 필드와 문서별 검증
- 거주자증명서와 제한세율 신청서의 성명, TIN, 거주국 교차검증

원본 저장소에는 세무 OCR용 Paddle 체크포인트가 포함되어 있지 않다. 런타임의 선택적 체크포인트 로더는 유지하지만, 운영 기본값은 원본과 같은 Tesseract 파이프라인이다. 존재하지 않는 가중치나 LLM 폴백으로 결과를 보정하지 않는다.

## 내부 API

- `POST /internal/v1/tax/documents/verify`: 문서 한 건의 OCR·필드·품질 검증
- `POST /internal/v1/tax/documents/compare`: 저장된 세 종류의 개별 검증 결과를 받아 원본 교차검증 규칙 실행

두 API 모두 Backend 전용 서비스 토큰이 필요하다. 개별 OCR의 원문은 요청 처리 중 임시 파일로만 생성하고 종료 시 삭제한다. 교차검증은 이미 저장된 정규화 필드와 개별 판정만 사용하므로 원문을 복호화·재전송·재OCR하지 않는다. 사용자 ID 대신 비가역 안전 식별자만 전달하며 OCR 결과를 정부 진위 확인으로 표현하지 않는다.

## 재현 검증

`scripts/verify_tax_bundle.py`에 거주자증명서, 아포스티유, 제한세율 신청서 경로를 순서대로 전달한다.

```shell
uv run python scripts/verify_tax_bundle.py residency.png apostille.png application.png
```

배포 이미지와 동일한 Tesseract·Poppler 환경에서 확인하려면 해당 스크립트를 AI 컨테이너 안에서 실행한다.
