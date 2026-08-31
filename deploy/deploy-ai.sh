#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEPLOY_ROOT=/opt/kmarket
readonly COMPOSE_FILE="$DEPLOY_ROOT/compose.prod.yaml"
readonly RUNTIME_ENV="$DEPLOY_ROOT/runtime.env"
readonly IMAGE_ENV="$DEPLOY_ROOT/image.env"
readonly HANA_RUNTIME="$DEPLOY_ROOT/hannah-runtime"
readonly HANA_REPOSITORY="https://github.com/Hana-harmony/Hana-Montana-AI.git"
readonly HANA_COMMIT="ab82ccc51cb096872f9a110a85c027a4158a147f"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "40자 Git 커밋 SHA가 필요합니다." >&2
  exit 2
fi
: "${GHCR_USERNAME:?required}"
: "${GHCR_TOKEN:?required}"

exec 9>"$DEPLOY_ROOT/.deploy.lock"
flock 9
umask 077

if [[ ! -d "$HANA_RUNTIME/.git" ]]; then
  echo "하나 모델 런타임 Git 저장소가 없습니다." >&2
  exit 1
fi
if ! git -C "$HANA_RUNTIME" diff --quiet || ! git -C "$HANA_RUNTIME" diff --cached --quiet; then
  echo "하나 모델 런타임에 추적 중인 로컬 변경이 있습니다." >&2
  exit 1
fi

# 공개 원격의 승인 커밋만 가져오고 핵심 모델·보고서 해시를 재검증한다.
git -C "$HANA_RUNTIME" fetch --depth=1 --no-tags "$HANA_REPOSITORY" "$HANA_COMMIT"
git -C "$HANA_RUNTIME" checkout --detach "$HANA_COMMIT"
if [[ "$(git -C "$HANA_RUNTIME" rev-parse HEAD)" != "$HANA_COMMIT" ]]; then
  echo "하나 모델 런타임 커밋 검증에 실패했습니다." >&2
  exit 1
fi
(
  cd "$HANA_RUNTIME"
  sha256sum --check --strict <<'CHECKSUMS'
04bb18037d28c59c487779531c90db5faa2e2136a3ca1dfe1d7af1a781ad6157  src/hannah_montana_ai/model_store/financial_nlp_ml.joblib
df852dcddb8e76436f415153fe34e86b9671bfc2134d78be648df513acb6f3f6  src/hannah_montana_ai/model_store/k_fnspid_impact_news_ml.joblib
a1b5a021ba47cff72300e77cf694cf3aa093b232efeecd9be14627ccb2e04822  src/hannah_montana_ai/model_store/k_fnspid_impact_disclosure_ml.joblib
c923702da9d221cd443dddc62df43c767c4cbbe851f249cc19b32f2fe5d016f6  reports/k-fnspid-impact-news-training-report.json
22a5eb0c47188d2b83e444b20dfa7854a79de883d8cb2726340d54409fa67a41  reports/k-fnspid-impact-disclosure-training-report.json
78c6db262e9263c84b32bd580c30b81335baea56ea210057fbb36edb58039a01  reports/kf-deberta-sentiment-training-report.json
996b6d0bcbd03a508dd36d7ceb2ab4135de1deaffa15854a987137147c5b71f9  reports/korean-finance-sentiment-benchmark.json
506a4290af390f9ebd3a3cabc8ae592e6c4c53837d44f1fb821c86819dd81c88  src/hannah_montana_ai/model_store/kf_deberta_sentiment/adapter_model.safetensors
CHECKSUMS
)

printf '%s' "$GHCR_TOKEN" | docker login ghcr.io --username "$GHCR_USERNAME" --password-stdin >/dev/null
unset GHCR_TOKEN

temporary=$(mktemp "$DEPLOY_ROOT/.image.env.XXXXXX")
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
docker image prune --force --filter until=168h >/dev/null
