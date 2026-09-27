# Business Entity Resolution — Amazon ML Challenge 2026 (team neural_nexus)

This pipeline links noisy Source 2 / Source 3 business records to Source 1 entities, optimised for per-entity
**macro F0.5**. Final submission: **public leaderboard 0.985252**, train out-of-fold macro F0.5 0.98962.
The methodology is described in `Documentation_template.md` at the top level of the submission zip.

```
raw TSV ─► normalise ─► candidates: TF-IDF/random-projection kNN (GPU) + exact keys ─► pre-ranker floor ══► candidate_pairs.tsv
        ─► round-1 GBDT ─► cross-encoder (uncertain band) ─► round-2 GBDT × 3 bags (consensus + competing clusters)
        ─► one owner per record ─► entity gate ─► per-S1 set rule tuned on out-of-fold F0.5 ══► matching_results.tsv
```

All settings of the final submission are in one file, **`configs/config.yaml`**.

---

## 1. Reproduce the final submission

### 1.1 Machine (for example on AWS)

| | Requirement |
|---|---|
| GPU | 1 or more NVIDIA GPUs with ≥ 16 GB, driver supporting CUDA 12 (e.g. `g5.4xlarge`, `g6.4xlarge`, `g4dn.12xlarge`; AWS Deep Learning AMIs qualify). Every multi-GPU path detects the GPUs itself; one GPU works. |
| RAM | ≥ 32 GiB, 64 GiB recommended (measured peak 30.3 GB, while training round 1 / round 2) |
| CPU | ≥ 4 cores; normalisation and pair features scale with cores |
| Disk | ~100 GB free for `work/` |
| Network | Once, to download `intfloat/multilingual-e5-small` (MIT, 118M parameters) from Hugging Face. No token, credentials or other network access. |
| Python | 3.12 |

Without a GPU the pipeline still runs (kNN and XGBoost fall back to CPU), but blocking takes many hours; see §7.

### 1.2 Setup

The submission zip's top level holds `output/`, `code/` and `Documentation_template.md`. Put the challenge data
next to them, and optionally the official validator:

```
.
├── code/business_entity_resolution/   (this folder)
├── dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
├── dataset/test/{test_source1,test_source2,test_source3}.tsv
├── utils/validate_submission.py       (optional: run automatically at the end)
└── output/                            (written by the pipeline)
```

```bash
cd code/business_entity_resolution
conda create -n ber python=3.12 -y && conda activate ber   # or any Python 3.12 environment (e.g. python3.12 -m venv)
pip install -r requirements.txt                             # (conda alternative: conda env create -f environment.yml)
export PYTHONPATH=$PWD/src                                  # the package is src/ber (no install step)
```

### 1.3 Run

```bash
python -m ber.pipeline.run --config configs/config.yaml --stage all
```

- Writes `../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv`, then runs the official
  validator (if present), then the error analysis.
- **Resumable:** every stage writes a completion marker, and the long stages save finished pieces. After an
  interruption, run the same command again; finished work is skipped.
- A quick end-to-end check on a small sample takes a few minutes (§2); worth doing first.

### 1.4 Outputs and reports

- `../../output/matching_results.tsv`: the scored file.
- `../../output/candidate_pairs.tsv`: the exact candidate set the matching models score (after the pre-ranker
  floor). Every match is one of the candidates by construction.
- `work/models/oof_report.json`: train out-of-fold macro F0.5 (by country, singleton / non-singleton), precision,
  recall and the candidate ceiling (best macro F0.5 achievable from the candidates).
- `work/models/error_analysis.md`: missed and wrong pairs by cause, loss by entity pattern.
- `work/models/thresholds.json`, `work/models/r2_bag_metrics.json`, `work/models/*/oof_metrics.json`,
  `work/models/*/importance.json`, `work/models/prerank/floor.json`, `work/models/prerank/expand.json`.

### 1.5 Runtime

Measured on Kaggle (4 CPU cores, 2× T4, 30 GB RAM), in minutes:

| normalise etc. | block | prerank + expand | features | r1 | cross-encoder | r2 | r2bag | gate, tune, predict, outputs, errors |
|---|---|---|---|---|---|---|---|---|
| ~25 (estimate) | 157 | 60 | 176 | 74 | 117 | 37 | 55 | 26 |

About 12 h in total. `features` and `normalize` are CPU-bound and shrink with more cores; `block` and the
cross-encoder with a faster GPU.

---

## 2. Quick check on a sample (a few minutes)

```bash
bash scripts/smoke_test.sh          # CE=0 bash scripts/smoke_test.sh  skips the cross-encoder (no model download)
```

It builds `sample_data/` from the training data once (3k train S1 + 2k pseudo-test S1, where about 30% of the US
clusters are relabelled "France" to exercise the path for a country without labels), runs every stage of
`configs/config.yaml` with sample-size overrides into `work_smoke/` and `output_smoke/`, validates the outputs and
scores them against the sample truth. Sample scores are far higher than real ones (sample distractors are random),
so it proves the pipeline runs, not its accuracy.

---

## 3. Inference only, with the trained models

The trained models (`neural_nexus_models.tar.gz`, 814 MB: mined tables, all GBDT fold models, the cross-encoder,
calibrators, floors and thresholds) are available on request.

```bash
mkdir -p work && tar xzf neural_nexus_models.tar.gz -C work     # tables.pkl + models/
python -m ber.pipeline.run --config configs/config.yaml --set run.inference_only=true --stage all
```

This processes `../../dataset/test/` only. Verified on sample data: it reproduces the full run's outputs byte for
byte.

---

## 4. Stage by stage

```bash
R="python -m ber.pipeline.run --config configs/config.yaml"
$R --stage ingest,eda,mine,normalize        # load TSVs, mine lookup tables from train pairs, parse every record
$R --stage block                            # candidate union: kNN (GPU) + exact keys; logs the train union ceiling
$R --stage prerank,expand                   # out-of-fold pre-ranker, floor tuned on the ceiling, 2-hop check
$R --stage features                         # round-1 pair features (sharded, all cores)
$R --stage r1                               # round-1 GBDT, out-of-fold on train
$R --stage ce_train,ce_infer                # cross-encoder: two cross-fitted halves, uncertain band only
$R --stage r2,r2bag                         # round 2 (+ 2 more bags on other entity samples, averaged)
$R --stage gate,tune,predict,outputs        # entity gate, thresholds, submission files + validator
$R --stage errors                           # train out-of-fold error analysis
```

- `--force` re-runs finished stages; `--from <stage>` runs from a stage onwards.
- `--set key.sub=value` overrides any config value.
- `--split train|test` runs a stage for one split; `ce_infer --shard i/n` splits cross-encoder scoring across
  machines.
- Moving between machines: copy `work/` (for example `aws s3 sync work/ s3://<bucket>/work/`).

---

## 5. Layout

```
code/business_entity_resolution/
├── README.md
├── requirements.txt             pinned pip environment
├── environment.yml              the same as a conda environment ("ber")
├── configs/config.yaml          all settings of the final submission
├── scripts/
│   ├── smoke_test.sh            quick end-to-end check on a sample
│   ├── make_sample.py           builds sample_data/ from the training data
│   ├── score_sample.py          scores sample outputs against the sample truth
│   └── package_submission.sh    builds <team>_submission.zip (+ optional models bundle)
└── src/ber/
    ├── io.py                    TSV → Parquet, unified id space, output writers
    ├── normalize/               text cleanup, Indic romaniser, consonant skeletons, country profiles, parsers
    ├── mining/tables.py         abbreviation / alias / component / affix tables mined from train pairs
    ├── blocking/                TF-IDF random-projection vectors, GPU/CPU kNN, key blocks, union, pre-ranker, 2-hop
    ├── features/                round-1 pair features, diff signatures, context, round-2 consensus and
    │                            competing-cluster features, admin-level detection (admin.py)
    ├── models/                  GBDT wrapper + out-of-fold training, streaming, calibration, cross-encoder, rounds, gate
    ├── decision/                vectorised per-S1 set selection, exact expected-F0.5 rule, threshold tuning
    ├── eval/                    exact metric and candidate ceiling, error analysis, diagnostics
    ├── eda/profile.py           data profile (eda stage)
    ├── checkpoint.py            optional Hugging Face mirror of work/ (off; used for resumable cloud sessions)
    ├── progress.py              pipeline progress bar and time estimate
    └── pipeline/                CLI (run.py), output writing and validation
```

**Memory model:** no stage holds the whole pair table in RAM. Candidates are written per country, the pre-ranker
scores them in chunks, round-1 features live in shards of contiguous S1 ranges, and each GBDT trains on the rows of
a sample of whole entities (`*.sample_entities`) and then scores every pair shard by shard (train rows
out-of-fold). Peak RAM depends on the training sample, not on the total number of pairs.

---

## 6. Configuration (`configs/config.yaml`)

| Setting | Final value | Meaning |
|---|---|---|
| `blocking.rp_dim`, `blocking.a1/n1` | 192; k 30/3, 20/3 | TF-IDF projection size; neighbours per S1 / per record in the address and name+address channels |
| `blocking.keyblock` | ≤ 50 S1, ≤ 5,000 pairs | exact-key block caps (generic names are dropped) |
| `prerank.ceiling_tol` | 0.0002 | allowed loss of the train candidate ceiling when choosing the pre-ranker floor |
| `expand` | self-gated | 2-hop expansion, used only if it raises the train ceiling by ≥ 0.0005 (it did not) |
| `r1.sample_entities` | 800k | training entities of round 1 |
| `r2.sample_entities`, `r2.bags` | 700k, 3 | training entities per round-2 bag; number of bags averaged |
| `ce.model_name`, `ce.band` | e5-small, [0.05, 0.95] | cross-encoder and the round-1 score band it scores |
| `features.admin_strip` | on, `pairs: false` | address admin level of countries without labels, ignored in round-2 consensus only |
| `decision.grid`, `decision.expected` | | search space of the per-country set rules |
| `decision.overrides.france` | κ 1.2 | France's set rule (no labels; see the documentation) |
| `*.gbdt` | XGBoost, 4 folds | backend (`xgboost` / `lightgbm`), device (`auto` / `cpu` / `cuda`), rounds, parameters |
| `run.max_gpus` | 0 | 0 = use all GPUs; 1 = single-GPU behaviour everywhere |
| `run.n_workers` | -1 | CPU workers (-1 = all cores) |
| `checkpoint.enabled` | false | optional private Hugging Face mirror of `work/` |

---

## 7. Hardware notes

- **GPUs:** the kNN search copies the index to every GPU and shares out query chunks (exact top-k, identical to a
  single-GPU result); XGBoost fold models train concurrently, one per GPU; the two cross-encoder halves run as one
  process per GPU. Each path falls back to one GPU or the sequential path if a worker fails.
  `--set run.max_gpus=1` forces single-GPU behaviour.
- **No GPU:** `--set blocking.knn.device=cpu` (multithreaded matmul, many hours) and XGBoost on CPU (automatic). The
  cross-encoder on CPU is slow; `--set ce.enabled=false` skips it at some accuracy cost.
- **Less RAM:** lower `r1/r2.sample_entities` and `max_train_rows`, `prerank.max_train_rows` and
  `features.join_rows`. This changes the trained models slightly.

---

## 8. Reproducibility, fair play and licences

- **Deterministic:** seeded RNGs; every model input and ranking sorted by `(s1_uid, rec_uid)` with explicit
  tie-breaks; cuBLAS pinned with `CUBLAS_WORKSPACE_CONFIG`. Two from-scratch sample runs give byte-identical
  outputs. GPU floating point can still differ in the last digits across GPU models.
- **No external data:** lookup tables are mined from training pairs only (each entry supported by ≥ 20 distinct S1
  entities); hand-written rules are abbreviation knowledge only (street types, legal forms); the address admin
  level of a country without labels is detected from that split's own addresses (`features/admin.py`). No
  gazetteers, geocoders, APIs or business databases.
- **Models:** XGBoost trees; `intfloat/multilingual-e5-small` (MIT, 118M parameters), fine-tuned on the training
  data only. Far below the 8B limit.

---

## 9. How the submitted files were produced

The submitted files were computed on Kaggle (2× T4, 12 h sessions) in checkpointed steps: each step restored the
finished stages of the previous ones and re-did the rest. `configs/config.yaml` merges the settings of the final
chain into one from-scratch run.

| Step | Change | Train out-of-fold | Leaderboard |
|---|---|---|---|
| Full run | whole pipeline, round 1 / 2 on 500k entities | 0.98933 | 0.984819 |
| A | round 1 re-trained on 800k entities | 0.98952 | — |
| G | round 2 with competing-cluster features, 700k entities | 0.98959 | 0.985166 |
| F | France admin level also ignored in round-1 features, France list sizes rescaled | same as G | 0.985031 (rejected) |
| **H** | **round-2 bagging (3 samples), on top of G** | **0.98962** | **0.985252 (final)** |

On sample data, this chain and one from-scratch run of the same settings give byte-identical outputs. On the full
data, the one difference is that the chain reused the cross-encoder scores of the full run, whose score band came
from its 500k-entity round 1. The Kaggle driver notebook is not part of this package (it only called this pipeline).

---

## 10. Troubleshooting

- **Out of memory:** see §7 "Less RAM". Changing `prerank.chunk_rows` or `features.shard_rows` discards that stage's
  saved partial progress (they are part of its resume plan).
- **Slow blocking:** check that the log shows `knn: ... on cuda`; `blocking.knn.mem_gb` sets the GPU memory used per
  query chunk.
- **`No module named ber`:** run `export PYTHONPATH=$PWD/src` from `code/business_entity_resolution/`.
- **Cross-encoder download fails:** the machine needs internet access to huggingface.co once (or a pre-downloaded
  model: `--set ce.model_name=/path/to/multilingual-e5-small`).
- **Validator skipped:** it is looked up at `../../utils/validate_submission.py`; set `paths.validator` otherwise.
