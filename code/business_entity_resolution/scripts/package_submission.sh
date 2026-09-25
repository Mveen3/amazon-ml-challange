#!/usr/bin/env bash
# Build <team>_submission.zip in the layout the challenge requires:
#   output/{matching_results.tsv,candidate_pairs.tsv}
#   code/business_entity_resolution/{src,configs,scripts,kaggle,README.md,requirements*.txt,environment.yml}
#   Documentation_template.md
# and (optionally) a models bundle for the inference-only reproduction path.
#
#   bash scripts/package_submission.sh <team_name> [--with-models]
#
# Environment overrides (used on Kaggle, where data and work dirs live on scratch disk):
#   WORK_DIR  pipeline work dir holding tables.pkl + models/  (default: <pkg>/work)
#   DATA_DIR  competition data root with test/                 (default: <project>/dataset)
#   OUT_DIR   where the zip / bundle are written               (default: <project>)
#   RESULTS_DIR  folder with the two submission TSVs            (default: <project>/output)
set -euo pipefail
TEAM="${1:?usage: package_submission.sh <team_name> [--with-models]}"
PKG="$(cd "$(dirname "$0")/.." && pwd)"          # code/business_entity_resolution
ROOT="$(cd "$PKG/../.." && pwd)"                 # project root (dataset/, output/, docs/, utils/)
WORK="${WORK_DIR:-$PKG/work}"
DATA="${DATA_DIR:-$ROOT/dataset}"
OUT="${OUT_DIR:-$ROOT}"
RESULTS="${RESULTS_DIR:-$ROOT/output}"
STAGE="$(mktemp -d)"
DEST="$STAGE/${TEAM}_submission"
ZIPCODE="$DEST/code/business_entity_resolution"

mkdir -p "$DEST/output" "$ZIPCODE"
cp "$RESULTS/matching_results.tsv" "$RESULTS/candidate_pairs.tsv" "$DEST/output/"
( cd "$PKG" && cp -r src configs scripts kaggle README.md requirements.txt requirements-kaggle.txt environment.yml \
      "$ZIPCODE/" )
find "$DEST/code" -name "__pycache__" -type d -prune -exec rm -rf {} +
DOC="$ROOT/docs/Documentation_template.md"
[ -f "$DOC" ] || DOC="$ROOT/docs/Documentation Template.md"
cp "$DOC" "$DEST/Documentation_template.md"

python3 "$ROOT/utils/validate_submission.py" --matching "$DEST/output/matching_results.tsv" \
    --candidate "$DEST/output/candidate_pairs.tsv" --test-dir "$DATA/test"

# python's zipfile instead of the zip binary (not installed everywhere, e.g. some Kaggle images)
( cd "$STAGE" && python3 -m zipfile -c "${TEAM}_submission.zip" "${TEAM}_submission" )
mkdir -p "$OUT"
mv "$STAGE/${TEAM}_submission.zip" "$OUT/"
echo "wrote $OUT/${TEAM}_submission.zip"

if [ "${2:-}" = "--with-models" ]; then
  # everything the inference-only path needs (tables + all trained models/thresholds)
  ( cd "$WORK" && tar czf "$OUT/${TEAM}_models.tar.gz" tables.pkl models )
  echo "wrote $OUT/${TEAM}_models.tar.gz"
fi
rm -rf "$STAGE"
