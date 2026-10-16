#!/usr/bin/env bash
# Fetch COCO 2017 into data/raw/coco.
#
# The architecture calls for ingesting instances_train2017.json in full: the
# instance segmentations give the ground-truth object inventory (which CHAIR and
# POPE score against) and the bounding boxes that Axis 2's low-resource crop is
# keyed to.
#
# Downloads are resumable and integrity-verified (see fetch.sh) — a truncated
# zip from a dropped connection is retried, not handed to the build.
#
# Total: ~20 GB. Re-running is safe — verified files are skipped.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${GIB_DATA_DIR:-$HERE/../data}/raw/coco"
mkdir -p "$ROOT"
cd "$ROOT"

fetch() { bash "$HERE/fetch.sh" "$1" "$2"; }

unpack() {
  local zip="$1" marker="$2"
  if [ -e "$marker" ]; then echo "  [skip] $marker"; return; fi
  echo "  [unzip] $zip"
  unzip -q -n "$zip"
}

echo "[coco] downloading to $ROOT"
fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" annotations.zip
fetch "http://images.cocodataset.org/zips/train2017.zip" train2017.zip
fetch "http://images.cocodataset.org/zips/val2017.zip"   val2017.zip

unpack annotations.zip annotations
unpack train2017.zip   train2017
unpack val2017.zip     val2017

echo "[coco] done. Verify:"
echo "  ls $ROOT/annotations/instances_train2017.json"
