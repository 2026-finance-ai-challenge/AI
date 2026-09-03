import ast
import runpy
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
validate_config = runpy.run_path(str(DEPLOY / "verify-ai-deployment.py"))["validate_config"]


def config(mounts: list[dict], *, read_only: bool = True) -> dict:
    return {
        "services": {"ai-api": {"read_only": read_only, "volumes": mounts}},
        "volumes": {"model-cache": {}},
    }


def test_missing_cache_is_rejected_before_service_replacement():
    with pytest.raises(ValueError, match="기존 AI 서비스는 변경하지 않았습니다"):
        validate_config(config([]))


@pytest.mark.parametrize(
    "mount",
    [
        {"type": "bind", "source": "/srv/cache"},
        {"type": "volume", "source": "model-cache", "read_only": True},
        {"type": "volume", "source": "undefined-cache"},
    ],
)
def test_invalid_cache_is_rejected(mount):
    with pytest.raises(ValueError):
        validate_config(config([{**mount, "target": "/home/kmarket/.cache"}]))


def test_valid_shared_cache_is_accepted_without_weakening_read_only():
    mounts = [{"type": "volume", "source": "model-cache", "target": "/home/kmarket/.cache"}]
    validate_config(config(mounts))
    with pytest.raises(ValueError):
        validate_config(config(mounts, read_only=False))


def test_preflight_precedes_mutation_and_is_uploaded_with_script():
    script = (DEPLOY / "deploy-ai.sh").read_text()
    assert script.index('| python3 "$DEPLOY_ROOT/verify-ai-deployment.py"') < script.index(
        "image_env_backup=$(mktemp"
    )
    workflow = (DEPLOY.parent / ".github/workflows/ci.yml").read_text()
    assert "deploy/deploy-ai.sh deploy/verify-ai-deployment.py" in workflow


def test_preflight_supports_host_python_not_only_container_python():
    ast.parse((DEPLOY / "verify-ai-deployment.py").read_text(), feature_version=(3, 10))
