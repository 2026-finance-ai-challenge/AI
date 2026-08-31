#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEPLOY_ROOT=/opt/kmarket
readonly COMPOSE_FILE="$DEPLOY_ROOT/compose.prod.yaml"
readonly RUNTIME_ENV="$DEPLOY_ROOT/runtime.env"
readonly IMAGE_ENV="$DEPLOY_ROOT/image.env"
readonly MODEL_RUNTIME="$DEPLOY_ROOT/kmarket-model-runtime"
readonly MODEL_SOURCE_REPOSITORY="https://github.com/Hana-harmony/Hana-Montana-AI.git"
readonly MODEL_SOURCE_COMMIT="ab82ccc51cb096872f9a110a85c027a4158a147f"
readonly MODEL_BASE="artifacts/pretraining/kf-deberta-k-fnspid-v4-dapt-temporal-v2/merged_fp32"
readonly LEGACY_RUNTIME="$DEPLOY_ROOT/hannah-runtime"

runtime_temporary=""
runtime_backup=""
image_env_backup=""
runtime_swapped=0

cleanup() {
  local status=$?
  trap - EXIT

  if (( status != 0 && runtime_swapped == 1 )); then
    local failed_runtime="$DEPLOY_ROOT/.kmarket-model-runtime.failed.$(date +%s)"
    if [[ -d "$MODEL_RUNTIME" ]]; then
      mv "$MODEL_RUNTIME" "$failed_runtime"
    fi
    if [[ -d "$runtime_backup" ]]; then
      mv "$runtime_backup" "$MODEL_RUNTIME"
    fi
    if [[ -f "$image_env_backup" ]]; then
      cp "$image_env_backup" "$IMAGE_ENV"
      chmod 600 "$IMAGE_ENV"
    fi

    # 실패한 컨테이너가 새 런타임을 계속 참조하지 않도록 기존 배포를 다시 기동한다.
    docker compose --profile worker --env-file "$RUNTIME_ENV" --env-file "$IMAGE_ENV" \
      -f "$COMPOSE_FILE" up -d --wait --wait-timeout 900 ai-api rag-worker backend || true
  fi

  if [[ -n "$runtime_temporary" && -d "$runtime_temporary" ]]; then
    rm -rf "$runtime_temporary"
  fi
  if [[ -n "$image_env_backup" && -f "$image_env_backup" ]]; then
    rm -f "$image_env_backup"
  fi
  exit "$status"
}
trap cleanup EXIT

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "40자 Git 커밋 SHA가 필요합니다." >&2
  exit 2
fi
: "${GHCR_USERNAME:?required}"
: "${GHCR_TOKEN:?required}"

exec 9>"$DEPLOY_ROOT/.deploy.lock"
flock 9
umask 077

if [[ ! -e "$MODEL_RUNTIME" && -d "$LEGACY_RUNTIME" && ! -L "$LEGACY_RUNTIME" ]]; then
  mv "$LEGACY_RUNTIME" "$MODEL_RUNTIME"
  ln -s "$(basename "$MODEL_RUNTIME")" "$LEGACY_RUNTIME"
fi
if [[ ! -d "$MODEL_RUNTIME/$MODEL_BASE" ]]; then
  echo "기존 K-Market 모델 번들의 기반 모델이 없습니다." >&2
  exit 1
fi
if [[ -e "$DEPLOY_ROOT/.kmarket-model-runtime.previous" ]]; then
  echo "이전 K-Market 모델 런타임 백업이 남아 있습니다." >&2
  exit 1
fi

# 기반 모델은 기존 서버 자산을 사용하므로 복사 전에도 무결성을 확인한다.
(
  cd "$MODEL_RUNTIME"
  sha256sum --check --strict <<'CHECKSUMS'
1f465cfdcef6a346e9fcce4071934d02a5007561a41ac551c5dc6d5f3f3cbd87  artifacts/pretraining/kf-deberta-k-fnspid-v4-dapt-temporal-v2/merged_fp32/model.safetensors
05751a3bc3e907f9d7c92c93517b68416496febc167db9a7363dc729d809675c  artifacts/pretraining/kf-deberta-k-fnspid-v4-dapt-temporal-v2/merged_fp32/config.json
CHECKSUMS
)

# 대용량 학습 저장소에서 운영에 필요한 파일만 고정 커밋으로 받는다.
runtime_temporary=$(mktemp -d "$DEPLOY_ROOT/.kmarket-model-runtime.XXXXXX")
git clone --filter=blob:none --no-checkout --depth=1 --branch main \
  "$MODEL_SOURCE_REPOSITORY" "$runtime_temporary"
git -C "$runtime_temporary" sparse-checkout init --no-cone
git -C "$runtime_temporary" sparse-checkout set --no-cone --stdin <<'SPARSE_PATHS'
/src/hannah_montana_ai/__init__.py
/src/hannah_montana_ai/core/
/src/hannah_montana_ai/domain/
/src/hannah_montana_ai/services/
/src/hannah_montana_ai/training/
/src/hannah_montana_ai/model_store/financial_nlp_ml.joblib
/src/hannah_montana_ai/model_store/k_fnspid_impact_news_ml.joblib
/src/hannah_montana_ai/model_store/k_fnspid_impact_disclosure_ml.joblib
/src/hannah_montana_ai/model_store/kf_deberta_sentiment/
/reports/k-fnspid-impact-news-training-report.json
/reports/k-fnspid-impact-disclosure-training-report.json
/reports/kf-deberta-sentiment-training-report.json
/reports/korean-finance-sentiment-benchmark.json
SPARSE_PATHS
git -C "$runtime_temporary" checkout --detach "$MODEL_SOURCE_COMMIT"
if [[ "$(git -C "$runtime_temporary" rev-parse HEAD)" != "$MODEL_SOURCE_COMMIT" ]]; then
  echo "K-Market 모델 번들 커밋 검증에 실패했습니다." >&2
  exit 1
fi

mkdir -p "$runtime_temporary/$(dirname "$MODEL_BASE")"
cp -a "$MODEL_RUNTIME/$MODEL_BASE" "$runtime_temporary/$MODEL_BASE"

# 교체 전 새 런타임의 코드, 분류 모델, 기반 모델을 모두 검증한다.
(
  cd "$runtime_temporary"
  sha256sum --check --strict <<'CHECKSUMS'
04bb18037d28c59c487779531c90db5faa2e2136a3ca1dfe1d7af1a781ad6157  src/hannah_montana_ai/model_store/financial_nlp_ml.joblib
df852dcddb8e76436f415153fe34e86b9671bfc2134d78be648df513acb6f3f6  src/hannah_montana_ai/model_store/k_fnspid_impact_news_ml.joblib
a1b5a021ba47cff72300e77cf694cf3aa093b232efeecd9be14627ccb2e04822  src/hannah_montana_ai/model_store/k_fnspid_impact_disclosure_ml.joblib
c923702da9d221cd443dddc62df43c767c4cbbe851f249cc19b32f2fe5d016f6  reports/k-fnspid-impact-news-training-report.json
22a5eb0c47188d2b83e444b20dfa7854a79de883d8cb2726340d54409fa67a41  reports/k-fnspid-impact-disclosure-training-report.json
78c6db262e9263c84b32bd580c30b81335baea56ea210057fbb36edb58039a01  reports/kf-deberta-sentiment-training-report.json
996b6d0bcbd03a508dd36d7ceb2ab4135de1deaffa15854a987137147c5b71f9  reports/korean-finance-sentiment-benchmark.json
506a4290af390f9ebd3a3cabc8ae592e6c4c53837d44f1fb821c86819dd81c88  src/hannah_montana_ai/model_store/kf_deberta_sentiment/adapter_model.safetensors
1f465cfdcef6a346e9fcce4071934d02a5007561a41ac551c5dc6d5f3f3cbd87  artifacts/pretraining/kf-deberta-k-fnspid-v4-dapt-temporal-v2/merged_fp32/model.safetensors
05751a3bc3e907f9d7c92c93517b68416496febc167db9a7363dc729d809675c  artifacts/pretraining/kf-deberta-k-fnspid-v4-dapt-temporal-v2/merged_fp32/config.json
CHECKSUMS
)

runtime_backup="$DEPLOY_ROOT/.kmarket-model-runtime.previous"
mv "$MODEL_RUNTIME" "$runtime_backup"
mv "$runtime_temporary" "$MODEL_RUNTIME"
runtime_temporary=""
runtime_swapped=1

printf '%s' "$GHCR_TOKEN" | docker login ghcr.io --username "$GHCR_USERNAME" --password-stdin >/dev/null
unset GHCR_TOKEN

temporary=$(mktemp "$DEPLOY_ROOT/.image.env.XXXXXX")
image_env_backup=$(mktemp "$DEPLOY_ROOT/.image.env.previous.XXXXXX")
cp "$IMAGE_ENV" "$image_env_backup"
awk -F= '$1 != "AI_IMAGE_TAG" { print }' "$IMAGE_ENV" >"$temporary"
printf 'AI_IMAGE_TAG=%s\n' "$1" >>"$temporary"
chmod 600 "$temporary"
mv "$temporary" "$IMAGE_ENV"

cd "$DEPLOY_ROOT"
docker compose --profile worker --env-file "$RUNTIME_ENV" --env-file "$IMAGE_ENV" -f "$COMPOSE_FILE" pull ai-api rag-worker
docker compose --profile worker --env-file "$RUNTIME_ENV" --env-file "$IMAGE_ENV" -f "$COMPOSE_FILE" up -d --wait --wait-timeout 900 ai-api rag-worker backend
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:15102/actuator/health >/dev/null

# 건강 확인만으로는 지연 로딩되는 분류 모델을 검증할 수 없어 실제 내부 계약을 호출한다.
docker compose --profile worker --env-file "$RUNTIME_ENV" --env-file "$IMAGE_ENV" -f "$COMPOSE_FILE" exec -T ai-api python - <<'PY'
import json
import os
import urllib.error
import urllib.request

payload = json.dumps(
    {
        "title": "정기보고서 제출",
        "paragraphs": ["회사는 정기보고서를 제출했습니다."],
        "candidate_companies": ["KART"],
        "source_type": "DISCLOSURE",
    }
).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/internal/v1/news/signals",
    data=payload,
    headers={
        "Authorization": "Bearer " + os.environ["KMARKET_AI_SERVICE_TOKEN"],
        "Content-Type": "application/json",
        "Host": "localhost",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
except urllib.error.HTTPError as exception:
    raise SystemExit(f"AI 분류 계약 확인 실패: HTTP {exception.code}") from exception

required = {
    "event_type",
    "sentiment",
    "importance",
    "market_impact",
    "market_impact_importance",
    "market_impact_score",
    "model",
}
if not required.issubset(result):
    raise SystemExit("AI 분류 계약 확인 실패: 필수 응답 누락")
PY

# 분류 계약까지 통과한 뒤에만 이전 런타임을 제거한다.
rm -rf "$runtime_backup"
runtime_backup=""
runtime_swapped=0
docker image prune --force --filter until=168h >/dev/null
