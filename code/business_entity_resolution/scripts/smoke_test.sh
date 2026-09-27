#!/usr/bin/env bash
# Quick end-to-end check: every stage of configs/config.yaml on a small sample (3k train S1 + 2k pseudo-test S1),
# a few minutes on one GPU (CPU works too, slower). Scores are far higher than on the real data (sample distractors
# are random); it proves the pipeline runs, not its accuracy.
#
#   bash scripts/smoke_test.sh               # build sample_data/ (once), run all stages, score against the sample truth
#   CE=0 bash scripts/smoke_test.sh          # without the cross-encoder (no model download)
#   DATA=/path/to/dataset bash scripts/smoke_test.sh   # challenge data elsewhere (default: ../../dataset)
#
# Finished stages are skipped on a re-run (as in the full pipeline); delete work_smoke/ to start over.
set -euo pipefail
cd "$(dirname "$0")/.."                          # code/business_entity_resolution
export PYTHONPATH="$PWD/src"
DATA="${DATA:-../../dataset}"
STAGES="${STAGES:-all}"                          # e.g. STAGES=ingest,eda,mine,normalize to run a part

[ -f sample_data/test_truth.tsv ] || python scripts/make_sample.py --data "$DATA" --out sample_data

EXTRA=()
[ "${CE:-1}" = 0 ] && EXTRA+=(--set ce.enabled=false)

# sample-scale settings: minimum supports and block limits sized for 5k entities, small chunks and shards
python -m ber.pipeline.run --config configs/config.yaml --stage "$STAGES" \
  --set paths.data_dir=sample_data --set paths.work_dir=work_smoke --set paths.output_dir=output_smoke \
  --set mining.min_support=3 --set blocking.keyblock.admin_df_min=5 --set blocking.keyblock.admin_df_max=3000 \
  --set blocking.knn.mem_gb=0.05 --set prerank.chunk_rows=50000 --set features.shard_rows=3000 \
  --set features.join_rows=10000 --set ce.max_pos=3000 --set ce.infer_chunk=5000 \
  ${EXTRA[@]+"${EXTRA[@]}"}

if [ "$STAGES" = all ]; then
  python scripts/score_sample.py --out output_smoke --truth sample_data/test_truth.tsv \
      --s1 sample_data/test/test_source1.tsv
fi
