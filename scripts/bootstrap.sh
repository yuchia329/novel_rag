#!/usr/bin/env bash
# Set up a fresh clone: install dependencies, restore the git-ignored files
# (.env, data/, large eval dumps) from S3, and pre-download the local models.
#
#   git clone <repo> && cd novel_rag && scripts/bootstrap.sh
#
# Needs: uv, aws-cli with credentials for the ignored-files bucket.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NAME="$(basename "$ROOT")"
cd "$ROOT"

for tool in uv aws; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done

# iCloud-synced folders (Desktop/Documents) evict .venv files and make imports
# hang, so keep the virtualenv outside the repo there.
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  case "$ROOT" in
    "$HOME/Desktop"*|"$HOME/Documents"*|*"Mobile Documents"*)
      export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/$NAME" ;;
  esac
fi

echo "==> Installing dependencies${UV_PROJECT_ENVIRONMENT:+ into $UV_PROJECT_ENVIRONMENT}"
uv sync

echo "==> Restoring ignored files from S3"
"$ROOT/scripts/ignored-sync.sh" pull

echo "==> Downloading embedding + reranker models (Hugging Face cache)"
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
    print("  ", repo, snapshot_download(repo))
EOF

echo
echo "Ready."
if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  echo "Run commands with:  export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
fi
echo "Check:  uv run novel-rag status"
