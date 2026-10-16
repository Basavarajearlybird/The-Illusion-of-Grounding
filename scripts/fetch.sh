#!/usr/bin/env bash
# Robust resumable download with integrity verification.
#
# Over a flaky connection curl can exit 0 on a clean mid-stream close, leaving a
# truncated file. Trusting the exit code then hands a broken zip to the build.
# This helper resumes with `-C -` and only declares success once the file
# actually verifies: `unzip -t` for zips, or a size floor for plain files.
#
# Usage: fetch.sh <url> <out> [min_bytes]
set -uo pipefail

url="$1"; out="$2"; min_bytes="${3:-0}"
max_attempts=30

verify() {
  [ -f "$out" ] || return 1
  case "$out" in
    *.zip)
      unzip -t "$out" >/dev/null 2>&1 ;;
    *)
      local sz; sz=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null || echo 0)
      [ "$sz" -ge "$min_bytes" ] ;;
  esac
}

if verify; then
  echo "  [ok] $out already complete"
  exit 0
fi

for attempt in $(seq 1 "$max_attempts"); do
  echo "  [get $attempt/$max_attempts] $out  ($(ls -lh "$out" 2>/dev/null | awk '{print $5}' || echo 0) so far)"
  curl -fSL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$out" "$url" 2>&1 | tail -1
  if verify; then
    echo "  [ok] $out verified ($(ls -lh "$out" | awk '{print $5}'))"
    exit 0
  fi
  echo "  [retry] $out incomplete/corrupt, resuming in 5s..."
  sleep 5
done

echo "  [FAIL] $out did not verify after $max_attempts attempts" >&2
exit 1
