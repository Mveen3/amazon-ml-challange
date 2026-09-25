#!/usr/bin/env bash
# Build <team>_submission.zip in the layout the challenge requires:
#   output/{matching_results.tsv,candidate_pairs.tsv}
#   code/business_entity_resolution/{src,configs,scripts,README.md,requirements.txt,environment.yml}
#   Documentation_template.md
# and (optionally) a models bundle for the inference-only reproduction path.
#
#   bash scripts/package_submission.sh <team_name> [--with-models]
set -euo pipefail
TEAM="${1:?usage: package_submission.sh <team_name> [--with-models]}"
PKG="$(cd "$(dirname "$0")/.." && pwd)"          # code/business_entity_resolution
ROOT="$(cd "$PKG/../.." && pwd)"                 # project root (dataset/, output/, docs/, utils/)
STAGE="$(mktemp -d)"
DEST="$STAGE/${TEAM}_submission"
ZIPCODE="$DEST/code/business_entity_resolution"

mkdir -p "$DEST/output" "$ZIPCODE"
cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" "$DEST/output/"
( cd "$PKG" && cp -r src configs scripts README.md requirements.txt environment.yml "$ZIPCODE/" )
find "$DEST/code" -name "__pycache__" -type d -prune -exec rm -rf {} +
DOC="$ROOT/docs/Documentation_template.md"
[ -f "$DOC" ] || DOC="$ROOT/docs/Documentation Template.md"
cp "$DOC" "$DEST/Documentation_template.md"

python3 "$ROOT/utils/validate_submission.py" --matching "$DEST/output/matching_results.tsv" \
    --candidate "$DEST/output/candidate_pairs.tsv" --test-dir "$ROOT/dataset/test"

( cd "$STAGE" && zip -qr "${TEAM}_submission.zip" "${TEAM}_submission" )
mv "$STAGE/${TEAM}_submission.zip" "$ROOT/"
echo "wrote $ROOT/${TEAM}_submission.zip"

if [ "${2:-}" = "--with-models" ]; then
  # everything the inference-only path needs (tables + all trained models/thresholds)
  ( cd "$PKG/work" && tar czf "$ROOT/${TEAM}_models.tar.gz" tables.pkl models )
  echo "wrote $ROOT/${TEAM}_models.tar.gz (host it e.g. on S3 and link it from README)"
fi
rm -rf "$STAGE"
