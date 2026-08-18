class RagProviderError(Exception):
    """외부 모델 또는 저장소 호출 실패."""


class RagIntegrityError(Exception):
    """색인 또는 모델 응답 무결성 오류."""
