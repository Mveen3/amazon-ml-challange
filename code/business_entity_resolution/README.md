# Business Entity Resolution — Amazon ML Challenge 2026

This pipeline links noisy Source 2 / Source 3 business records to clean Source 1 entities. It is optimised for per-entity **macro F0.5**.

The design, the data facts and the reasoning behind every choice are in [`docs/Pipeline_Architecture.md`](../../docs/Pipeline_Architecture.md). This README explains **how to run it**.

```
raw TSV ─► 0 normalise ─► 1 candidates (TF-IDF/RP kNN + keys ─► pre-ranker floor ─► 2-hop) ══► candidate_pairs.tsv
        ─► 2 round-1 LightGBM ─► cross-encoder (uncertain band) ─► round-2 LightGBM (consensus + competition)
        ─► 3 one owner per record ─► entity gate ─► first-member rule + t_add(k) + caps ══► matching_results.tsv
```

---

## 1. Layout

```
code/business_entity_resolution/
├── README.md                    this file
├── requirements.txt             pinned (pip)
├── environment.yml              pinned (conda env "ml")
├── configs/
│   ├── default.yaml             full-scale settings (every knob documented inline)
│   └── smoke.yaml               tiny end-to-end test on sample_data/
├── scripts/
│   ├── make_sample.py           build sample_data/ from the training set
│   ├── score_sample.py          score smoke-test outputs against the hidden sample truth
│   └── package_submission.sh    build <team>_submission.zip (+ optional models bundle)
└── src/ber/
    ├── io.py                    TSV → Parquet, unified uid space, output writers
    ├── normalize/               text cleanup, Indic romaniser + skeleton, country profiles, name/address parsers
    ├── mining/tables.py         token / abbreviation / component / affix tables mined from train pairs
    ├── blocking/                RP-TF-IDF vectors, GPU/CPU kNN, key blocking, union, pre-ranker, 2-hop
    ├── features/                round-1 pair features, diff signatures, context, round-2 consensus
    ├── models/                  GBDT wrapper + OOF, calibration, cross-encoder, bi-encoder, rounds, gate
    ├── decision/                vectorised set selection and threshold tuning
    ├── eval/                    exact metric + ceiling, stress test, S1↔S1 diagnostic
    ├── eda/profile.py           recomputes the data facts of the architecture doc (§2)
    └── pipeline/                CLI (run.py) and output writing / validation
```

All intermediate artefacts go to `work/` (Parquet, `.npy` and model files). Each stage writes a completion marker, so the pipeline **resumes** where it stopped.

---

## 2. Environment

```bash
conda env create -f environment.yml        # creates the conda env "ml"
conda activate ml
cd code/business_entity_resolution
export PYTHONPATH=$PWD/src
```

The CPU pipeline was verified end to end with the pinned versions. The neural stages (cross-encoder, optional bi-encoder) need a modern `transformers` (pinned: 4.46.3). An environment with `transformers` 2.x cannot load current Hugging Face models; either upgrade it or set `ce.enabled: false`.

---

## 3. Smoke test (a few minutes on a laptop)

```bash
python scripts/make_sample.py --data ../../dataset --out sample_data     # 3k train S1 + 2k pseudo-test S1
python -m ber.pipeline.run --config configs/smoke.yaml --stage all        # every stage, CPU only
python scripts/score_sample.py                                            # macro F0.5 vs hidden sample truth
```

The pseudo-test split relabels about 30% of US clusters as "France", to exercise the path for a country with no training data. Its truth file is never read by the pipeline. Sample scores are much higher than real ones, because sample distractors are random rather than planted near-duplicates.

---

## 4. Full run on AWS

### 4.1 Machines

| Box | Suggested instance | Used for |
|---|---|---|
| CPU | `r7i.16xlarge` (64 vCPU, 512 GiB) or larger | every stage except the neural ones; the pre-ranker union for train is about 150–200M pairs |
| GPU | `g6e.2xlarge` (L40S 48 GB) or `p4d`/`p5` | kNN (much faster on GPU), cross-encoder train/infer, optional bi-encoder |

A single GPU box with enough RAM (for example `g6e.16xlarge`, 512 GiB) can run everything.

- **Request the GPU service quota first.** New accounts often start at zero for G/P instances, and approval can take hours.
- **Without a GPU:**
  - kNN falls back to multithreaded CPU matmul; set `blocking.knn.device: cpu`. This takes hours rather than minutes.
  - Set `ce.enabled: false`.

### 4.2 Commands

From `code/business_entity_resolution/`, with the challenge data in `../../dataset/{train,test}`:

```bash
python -m ber.pipeline.run --config configs/default.yaml --stage all
```

The same run, stage by stage (useful for monitoring and for moving between boxes):

```bash
R="python -m ber.pipeline.run --config configs/default.yaml"
$R --stage ingest,eda,mine,normalize        # CPU
$R --stage dense                            # no-op unless blocking.dense.enabled (GPU)
$R --stage block                            # GPU recommended (kNN); logs train union ceiling / pair recall
$R --stage prerank,expand                   # CPU: OOF pre-ranker, floor tuned on ceiling, 2-hop
$R --stage features                         # CPU: round-1 features (sharded, all cores)
$R --stage r1                               # CPU: OOF round-1 LightGBM
$R --stage ce_train                         # GPU: two cross-fitted halves
$R --stage ce_infer --split train --shard 0/2   # GPU: shard across GPUs/boxes if you like
$R --stage ce_infer --split train --shard 1/2
$R --stage ce_infer --split test  --shard 0/1
$R --stage r2,gate,tune,predict,outputs     # CPU: round 2, gate, thresholds, submission files
```

- **Moving work between boxes:** `aws s3 sync work/ s3://<bucket>/work/` on one box and the reverse on the other.
- **Re-running a stage:** add `--force`.
- **Continuing from a stage:** `--from <stage>`.
- **Changing a config value on the command line:** `--set key.sub=value`.

### 4.3 Outputs

- `../../output/matching_results.tsv` is the file that gets scored.
- `../../output/candidate_pairs.tsv` is the final candidate set. It is written *after* the 2-hop expansion, so every match is guaranteed to be one of the candidates.
- Both are checked with `utils/validate_submission.py` automatically.
- Reports:
  - `work/models/oof_report.json`: out-of-fold macro F0.5 by profile, singleton / non-singleton, true group size, best achievable score.
  - `work/models/*/oof_metrics.json` and `importance.json`
  - `work/models/prerank/floor.json` (best achievable score of the union vs chosen floor)
  - `work/models/prerank/expand.json` (2-hop gain)
  - `work/models/stress_check.json`
  - `work/models/thresholds.json`

---

## 5. Leaderboard probes

The doc's §12 plan is to spend submissions only on questions that validation cannot answer. Each probe writes to `../../output/probes/<name>/` and leaves the main submission untouched.

```bash
R="python -m ber.pipeline.run --config configs/default.yaml"
$R --stage outputs --probe all_empty                                            # public singleton rate
$R --stage predict,outputs --probe fr_empty  --set decision.overrides.france.empty=true
$R --stage predict,outputs --probe fr_strict --set decision.overrides.france.delta_shift=0.05
$R --stage predict,outputs --probe fr_loose  --set decision.overrides.france.delta_shift=-0.05
```

- **France score:** F_France ≈ s_F + (LB_main − LB_fr_empty) / w_F, where w_F is France's share of S1 entities (≈0.15) and s_F is its singleton rate (≈0.056).
- **Adopting a winning override:** put it under `decision.overrides` in the config and re-run `predict,outputs`.

---

## 6. Reproducibility and audit

- **Deterministic:**
  - Seeded RNGs; LightGBM runs with `deterministic` and `force_col_wise`.
  - Every model input and ranking is sorted by `(s1_uid, rec_uid)`, and all tie-breaks are explicit.
  - Two from-scratch smoke runs produce byte-identical `matching_results.tsv` and `candidate_pairs.tsv`.
  - cuBLAS is pinned with `CUBLAS_WORKSPACE_CONFIG`.
- **Inference-only path (no retraining):**
  1. Restore `work/tables.pkl` and `work/models/` from the models bundle (`bash scripts/package_submission.sh <team> --with-models` creates it).
  2. Run:
     ```bash
     python -m ber.pipeline.run --config configs/default.yaml --set run.inference_only=true --stage all
     ```
  - This processes the test split only and loads every model, calibrator, floor and threshold.
  - It was verified to reproduce the full run's outputs byte for byte.
- **No external data:**
  - Lookup tables are mined from training pairs only, and each entry must be supported by ≥ 20 distinct S1 entities.
  - Hand-written rules are abbreviation knowledge only (street types, legal forms).
  - No gazetteers, geocoders or APIs are used.
- **Model licences and size:**
  - LightGBM / XGBoost: trees only.
  - `BAAI/bge-reranker-v2-m3`: Apache-2.0, 568M parameters.
  - Optional `intfloat/multilingual-e5-base`: MIT, 278M parameters.
  - Total stays well under 8B parameters.

---

## 7. Packaging

```bash
bash scripts/package_submission.sh <team_name>              # -> ../../<team_name>_submission.zip
bash scripts/package_submission.sh <team_name> --with-models   # + ../../<team_name>_models.tar.gz
```

Fill in `docs/Documentation Template.md` before packaging; it is copied into the zip as `Documentation_template.md`.

The local layout (`code/business_entity_resolution/`) is the layout the challenge requires inside the zip, so the same paths work in both places.

---

## 8. Key configuration knobs (`configs/default.yaml`)

| Knob | Meaning |
|---|---|
| `blocking.a1/n1.k_fwd/k_rev` | Neighbours per S1 / per record in the address and name+address channels |
| `blocking.keyblock.*` | Exact-key block caps and the document-frequency band for locality tokens |
| `blocking.dense.enabled` | Optional bi-encoder channel (enable only if the best achievable score needs it) |
| `prerank.ceiling_tol` | Allowed best-achievable-score loss when choosing the pre-ranker floor (default 0.0002) |
| `expand.*` | 2-hop expansion; applied to test only if it raised the train best achievable score by `min_gain` |
| `r1/r2.monotone` | Monotone-constraint ablation for France robustness |
| `ce.*` | Cross-encoder model, band, batch sizes |
| `decision.caps` | Maximum S2 (5) / S3 (6) records per S1, verified on train |
| `decision.t_add_base` | Break-even thresholds for adding the k-th member (doc Appendix B) |
| `decision.grid` | Tuning grid for `kappa` (first-member rule), `t_min`, `delta` |
| `decision.fallback_profile` | Thresholds and calibration used for profiles with no training data (France) |
| `stress.*` | Unmatched-density stress test (drop 19% of train S1s) used for robust tuning |

---

## 9. Troubleshooting

- **Out of memory in `prerank`:** lower `prerank.chunk_rows` and/or `blocking.a1.k_fwd`.
- **Slow kNN:** check that the log line `knn: ... on cuda` appears; tune `blocking.knn.mem_gb` to fit your GPU.
- **Validator warning (a match outside the candidates):** this cannot happen by construction; `outputs` asserts it. If it appears, re-run `expand` and `features` so both files come from the same run.
- **LightGBM missing:** the wrapper falls back to XGBoost automatically (`gbdt_base.backend`).
