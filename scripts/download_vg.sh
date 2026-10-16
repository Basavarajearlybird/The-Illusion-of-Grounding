#!/usr/bin/env bash
# Fetch Visual Genome into data/raw/vg.
#
# VG supplies the attribute and relation sub-types: its scene graphs carry
# explicit (subject, predicate, object) triples and per-object attribute lists,
# which is what makes those questions machine-verifiable rather than
# LLM-asserted.
#
# VG images overlap heavily with COCO (image_data.json carries coco_id), which
# is exploited deliberately — a single image can carry all three sub-types, and
# the COCO instance inventory stays available for CHAIR and POPE on VG images.
#
# NOTE: the scene-graph JSON lives on the UW mirror, but the IMAGES live on the
# Stanford mirror (the UW image URLs 404). Both are wired below.
#
# Downloads are resumable and integrity-verified (see fetch.sh).
# Total: ~25 GB.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${GIB_DATA_DIR:-$HERE/../data}/raw/vg"
mkdir -p "$ROOT"
cd "$ROOT"

UW="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
STANFORD="https://cs.stanford.edu/people/rak248"

fetch() { bash "$HERE/fetch.sh" "$1" "$2"; }

unpack() {
  local zip="$1" marker="$2"
  if [ -e "$marker" ]; then echo "  [skip] $marker"; return; fi
  echo "  [unzip] $zip"
  unzip -q -n "$zip"
}

echo "[vg] downloading to $ROOT"
fetch "$UW/scene_graphs.json.zip"       scene_graphs.json.zip
fetch "$UW/image_data.json.zip"         image_data.json.zip
fetch "$STANFORD/VG_100K/images.zip"    images.zip
fetch "$STANFORD/VG_100K_2/images2.zip" images2.zip

unpack scene_graphs.json.zip scene_graphs.json
unpack image_data.json.zip   image_data.json
unpack images.zip            VG_100K
unpack images2.zip           VG_100K_2

echo "[vg] done. Verify:"
echo "  ls $ROOT/scene_graphs.json $ROOT/VG_100K | head"
