"""실행 중 서비스를 바꾸기 전에 AI 캐시 마운트 계약을 확인한다."""

import json
import sys


def validate_config(config: dict) -> None:
    service = config.get("services", {}).get("ai-api", {})
    cache = next(
        (
            mount
            for mount in service.get("volumes", [])
            if mount.get("target") == "/home/kmarket/.cache"
        ),
        None,
    )
    if (
        not service.get("read_only")
        or not cache
        or cache.get("type") != "volume"
        or cache.get("read_only", False)
        or cache.get("source") not in config.get("volumes", {})
    ):
        raise ValueError(
            "AI 캐시 마운트가 없습니다. Backend의 최신 운영 Compose를 먼저 배포하세요. "
            "기존 AI 서비스는 변경하지 않았습니다."
        )


if __name__ == "__main__":
    try:
        validate_config(json.load(sys.stdin))
    except (ValueError, TypeError, AttributeError):
        sys.exit(
            "AI 배포 사전 검증 실패: 읽기 전용 서비스의 쓰기 가능한 모델 캐시 볼륨이 필요합니다."
        )
