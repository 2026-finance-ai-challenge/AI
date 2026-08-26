#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEPLOY_ROOT=/opt/kmarket
readonly COMPOSE_FILE="$DEPLOY_ROOT/compose.prod.yaml"
readonly RUNTIME_ENV="$DEPLOY_ROOT/runtime.env"
readonly IMAGE_ENV="$DEPLOY_ROOT/image.env"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "40자 Git 커밋 SHA가 필요합니다." >&2
  exit 2
fi
: "${GHCR_USERNAME:?required}"
: "${GHCR_TOKEN:?required}"

exec 9>"$DEPLOY_ROOT/.deploy.lock"
flock 9
umask 077

printf '%s' "$GHCR_TOKEN" | docker login ghcr.io --username "$GHCR_USERNAME" --password-stdin >/dev/null
unset GHCR_TOKEN

temporary=$(mktemp "$DEPLOY_ROOT/.image.env.XXXXXX")
awk -F= '$1 != "AI_IMAGE_TAG" { print }' "$IMAGE_ENV" >"$temporary"
printf 'AI_IMAGE_TAG=%s\n' "$1" >>"$temporary"
chmod 600 "$temporary"
mv "$temporary" "$IMAGE_ENV"

cd "$DEPLOY_ROOT"
docker compose --profile worker --env-file "$RUNTIME_ENV" --env-file "$IMAGE_ENV" -f "$COMPOSE_FILE" pull ai-api rag-worker
docker compose --profile worker --env-file "$RUNTIME_ENV" --env-file "$IMAGE_ENV" -f "$COMPOSE_FILE" up -d --wait --wait-timeout 300 ai-api rag-worker backend frontend
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:15101/healthz >/dev/null
docker image prune --force --filter until=168h >/dev/null
