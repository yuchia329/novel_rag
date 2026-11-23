#!/usr/bin/env bash
# Push/pull the git-ignored files of this repo (secrets, data, indexes, large
# eval dumps) to/from S3, keeping the repo-relative layout.
#
#   s3://$IGNORED_BUCKET/<repo-name>/<same path as in the repo>
#
# Usage:
#   scripts/ignored-sync.sh list              # show what push would upload
#   scripts/ignored-sync.sh push [--delete] [--dryrun]
#   scripts/ignored-sync.sh pull [--dryrun]
#
# What is synced: every path `git` reports as ignored, minus rebuildable junk
# (virtualenvs, __pycache__, tool caches, .DS_Store, editor folders).
# `push --delete` also removes S3 objects inside synced folders that no longer
# exist locally. `pull` never deletes local files.
#
# Env overrides: IGNORED_BUCKET (default yuchia-repository-ignored),
#                IGNORED_PREFIX (default: repo folder name), AWS_PROFILE.
set -euo pipefail

BUCKET="${IGNORED_BUCKET:-yuchia-repository-ignored}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PREFIX="${IGNORED_PREFIX:-$(basename "$ROOT")}"
DEST="s3://${BUCKET}/${PREFIX}"

# Ignored paths that are rebuildable or machine-specific: never synced.
JUNK_RE='(^|/)(\.DS_Store|__pycache__/|\.pytest_cache/|\.ruff_cache/|\.mypy_cache/|\.venv/|venv/|node_modules/|[^/]*\.egg-info/|dist/|build/|\.vscode/|\.idea/)$'
# Same junk, as aws-cli filters for files inside synced folders.
AWS_EXCLUDES=(--exclude ".DS_Store" --exclude "*/.DS_Store" --exclude "*__pycache__/*")

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

ignored_paths() {
  # --directory collapses fully-ignored folders (data/) into one entry.
  git -C "$ROOT" ls-files --others --ignored --exclude-standard --directory |
    grep -Ev "$JUNK_RE" || true
}

cmd="${1:-}"; shift || true
DELETE=(); DRY=()
for a in "$@"; do
  case "$a" in
    --delete) DELETE=(--delete) ;;
    --dryrun|--dry-run) DRY=(--dryrun) ;;
    *) usage ;;
  esac
done

case "$cmd" in
  list)
    echo "Target: $DEST/"
    ignored_paths | while IFS= read -r p; do
      printf '%8s  %s\n' "$(du -sh "$ROOT/$p" 2>/dev/null | cut -f1)" "$p"
    done
    ;;

  push)
    echo "Pushing ignored files of $ROOT -> $DEST/"
    ignored_paths | while IFS= read -r p; do
      src="$ROOT/${p%/}"
      if [ -d "$src" ]; then
        aws s3 sync "$src" "$DEST/${p%/}" "${AWS_EXCLUDES[@]}" \
          --sse AES256 --no-progress ${DELETE[@]+"${DELETE[@]}"} ${DRY[@]+"${DRY[@]}"}
      elif [ -f "$src" ]; then
        aws s3 cp "$src" "$DEST/$p" --sse AES256 --no-progress ${DRY[@]+"${DRY[@]}"}
      fi
    done
    echo "Done."
    ;;

  pull)
    echo "Pulling $DEST/ -> $ROOT"
    aws s3 sync "$DEST" "$ROOT" "${AWS_EXCLUDES[@]}" --no-progress ${DRY[@]+"${DRY[@]}"}
    [ -f "$ROOT/.env" ] && chmod 600 "$ROOT/.env"
    echo "Done."
    ;;

  *) usage ;;
esac
