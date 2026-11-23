#!/usr/bin/env bash
# Fetch gemma-4-E2B-it.litertlm (2,588,147,712 bytes) on a mainland-China host.
# Primary path: hf-mirror + aria2. Fallbacks: hf-mirror single-stream, ModelScope.
set -euo pipefail

REPO="litert-community/gemma-4-E2B-it-litert-lm"
FILE="gemma-4-E2B-it.litertlm"
DEST="${DEST:-/opt/sign2speak/models}"
EXPECTED_SIZE=2588147712

mkdir -p "$DEST"
cd "$DEST"

if [[ -f "$FILE" && "$(stat -c%s "$FILE")" -eq "$EXPECTED_SIZE" ]]; then
  echo "model already present and size-checked: $DEST/$FILE"
  exit 0
fi

echo "==> aria2 via hf-mirror"
aria2c -x 16 -s 16 -k 1M --continue=true --file-allocation=none \
  --max-tries=8 --retry-wait=5 --summary-interval=30 \
  -d "$DEST" -o "$FILE" \
  "https://hf-mirror.com/${REPO}/resolve/main/${FILE}" || true

size=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
if [[ "$size" -ne "$EXPECTED_SIZE" ]]; then
  echo "==> aria2 incomplete ($size bytes); retry with hf CLI"
  HF_ENDPOINT=https://hf-mirror.com \
    uvx --from huggingface_hub hf download "$REPO" "$FILE" --local-dir "$DEST" || true
  size=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
fi

if [[ "$size" -ne "$EXPECTED_SIZE" ]]; then
  echo "ERROR: size mismatch after retries ($size != $EXPECTED_SIZE)" >&2
  echo "Try ModelScope or upload the file manually (see deploy/README.md)." >&2
  exit 1
fi

echo "OK: $(stat -c%s "$FILE") bytes"
sha256sum "$FILE" | tee MANIFEST.txt
echo "fetched_at=$(date -Is)" >> MANIFEST.txt
