#!/bin/bash
# Run wordmute in its container.
#   ./run.sh [--dry-run] [--replace-review] <video or folder>...
#   ./run.sh --build        build (or rebuild) the image
#   ./run.sh --test         run the test suite in the image
# Inputs are mounted read-only at their own paths, so originals cannot be modified.
# The data folder (output, reports, scratch; see config.toml) is the only writable media path.
# It is named by WORDMUTE_DATA, in the environment or in local.env beside this script (not in git).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/local.env" ]; then . "$HERE/local.env"; fi
IMAGE="${WORDMUTE_IMAGE:-cleanvid-wordmute:latest}"
DATA="${WORDMUTE_DATA:-}"
CONFIG="${WORDMUTE_CONFIG:-$HERE/config.toml}"
MODELS="${WORDMUTE_MODELS:-$HOME/.cache/cleanvid-wordmute/models}"

if [ "${1:-}" = "--build" ]; then
  exec docker build -t "$IMAGE" "$HERE"
fi
if [ "${1:-}" = "--test" ]; then
  shift
  exec docker run --rm --user "$(id -u):$(id -g)" --entrypoint python "$IMAGE" -m pytest -q -p no:cacheprovider "$@"
fi

if [ -z "$DATA" ]; then
  echo "Set WORDMUTE_DATA to the data folder: export it, or put WORDMUTE_DATA=/path in $HERE/local.env" >&2
  exit 1
fi
DATA="$(realpath -m "$DATA")"
mkdir -p "$DATA" "$MODELS"
mounts=(-e "WORDMUTE_DATA=$DATA" -v "$DATA:$DATA" -v "$MODELS:/models" -v "$CONFIG:/app/config.toml:ro")
args=()
for arg in "$@"; do
  if [ -e "$arg" ]; then
    path="$(realpath -s "$arg")"
    dir="$path"; [ -d "$path" ] || dir="$(dirname "$path")"
    mounts+=(-v "$dir:$dir:ro")
    args+=("$path")
  else
    args+=("$arg")
  fi
done

exec docker run --rm --user "$(id -u):$(id -g)" "${mounts[@]}" "$IMAGE" --config /app/config.toml "${args[@]}"
