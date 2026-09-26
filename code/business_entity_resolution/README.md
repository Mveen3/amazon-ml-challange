# Business Entity Resolution — Amazon ML Challenge 2026

This pipeline links noisy Source 2 / Source 3 business records to clean Source 1 entities. It is optimised for per-entity **macro F0.5**.

The design, the data facts and the reasoning behind every choice are in [`docs/Pipeline_Architecture.md`](../../docs/Pipeline_Architecture.md). This README explains **how to run it**.

```
raw TSV ─► 0 normalise ─► 1 candidates (TF-IDF/RP kNN + keys ─► pre-ranker floor ─► 2-hop) ══► candidate_pairs.tsv
        ─► 2 round-1 LightGBM ─► cross-encoder (uncertain band) ─► round-2 LightGBM (consensus + competition)
        ─► 3 one owner per record ─► entity gate ─► first-member rule + t_add(k) + caps ══► matching_results.tsv
```

---

## 0. Final submission: how to reproduce it (read this first)

The final submission is produced by **`configs/final.yaml`**: the Kaggle profile (2× T4, 4 CPU, 30 GB RAM; a
larger machine works unchanged) plus the final improvements, which are listed at the top of that file. The
methodology write-up is `docs/Documentation_template.md`.

**From scratch (train + predict), from this folder, with the challenge data in `../../dataset/{train,test}`:**
```bash
pip install -r requirements.txt            # or requirements-kaggle.txt on a Kaggle image
python -m ber.pipeline.run --config configs/final.yaml --stage all
```
- It writes `../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv` and runs the official
  validator on them.
- Runtime on Kaggle 2× T4 is about 11 h. Every stage resumes after an interruption (see §4.3.1).

**Inference only on new test data, with our trained models:**
```bash
mkdir -p work && tar xzf <team>_models.tar.gz -C work     # tables.pkl + models/ (incl. density_ref.json)
python -m ber.pipeline.run --config configs/final.yaml --set run.inference_only=true --stage all
```
This processes only `../../dataset/test/` and loads every model, calibrator, floor and threshold. Verified on
sample data: it reproduces the full run's outputs byte for byte.

**How the final files were computed on Kaggle (checkpointed tracks; same result as one from-scratch run):**
1. `full` = `configs/kaggle.yaml`: the full pipeline run.
2. `configs/track_a.yaml`: r1/r2 re-trained on 800k entities (starts from `r1`).
3. `configs/track_g.yaml`: final round 2 with competing-cluster features and per-country list sizes (starts from
   `r2`).
4. `configs/track_f.yaml`: inference-only, test features recomputed with the France adaptations and re-scored with
   the trained models (starts from `features`). Its outputs and models bundle are the final submission.

Tracks 3 and 4 run in one Kaggle session: notebook `CONFIG = "configs/track_g.yaml"`,
`THEN_CONFIG = "configs/track_f.yaml"`. Track G's files end up in `/kaggle/working/track_g/`, Track F's at the top
level.

On sample data this chain gives a `matching_results.tsv` / `candidate_pairs.tsv` byte-identical to one
from-scratch `final.yaml` run. On the full data the one difference is that the chain reuses the cross-encoder
scores of the first run (their score band came from the 500k-entity round 1).

---

## 1. Layout

```
code/business_entity_resolution/
├── README.md                    this file
├── requirements.txt             pinned (pip)
├── requirements-kaggle.txt      the few extra packages a Kaggle image needs
├── environment.yml              pinned (conda env "ml")
├── configs/
│   ├── default.yaml             full-scale settings (every knob documented inline)
│   ├── kaggle.yaml              Kaggle profile: 2x T4, 4 CPU cores, ~29 GB RAM, 12 h sessions
│   └── smoke.yaml               tiny end-to-end test on sample_data/
├── kaggle/
│   └── amazon_ml_kaggle.ipynb   Kaggle notebook: imports data + code, trains, writes the submission
├── scripts/
│   ├── make_sample.py           build sample_data/ from the training set
│   ├── score_sample.py          score smoke-test outputs against the hidden sample truth
│   ├── kaggle_prepare.py        find/unpack the Kaggle dataset and link it into the expected layout
│   └── package_submission.sh    build <team>_submission.zip (+ optional models bundle)
└── src/ber/
    ├── io.py                    TSV → Parquet, unified uid space, output writers
    ├── normalize/               text cleanup, Indic romaniser + skeleton, country profiles, name/address parsers
    ├── mining/tables.py         token / abbreviation / component / affix tables mined from train pairs
    ├── blocking/                RP-TF-IDF vectors, GPU/CPU kNN, key blocking, union, pre-ranker, 2-hop
    ├── features/                round-1 pair features, diff signatures, context, round-2 consensus
    ├── models/                  GBDT wrapper + OOF, out-of-core streaming, calibration, cross-encoder,
    │                            bi-encoder, rounds, gate
    ├── decision/                vectorised set selection and threshold tuning
    ├── eval/                    exact metric + ceiling, stress test, S1↔S1 diagnostic
    ├── eda/profile.py           recomputes the data facts of the architecture doc (§2)
    └── pipeline/                CLI (run.py) and output writing / validation
```

All intermediate artefacts go to `work/` (Parquet, `.npy` and model files). Each stage writes a completion marker, so the pipeline **resumes** where it stopped.

**Memory model.** No stage holds the whole pair table in RAM:
- Candidates are written per country.
- The pre-ranker scores them in chunks.
- Round-1 features live in shards of contiguous S1 ranges.
- Each GBDT trains on the rows of a random sample of whole entities (`*.sample_entities`), then scores every pair shard by shard. Train rows are scored out-of-fold.

Peak RAM therefore depends on the largest country partition and the training sample, not on the total number of pairs.

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

## 4. Full run on Kaggle (2× T4)

`kaggle/amazon_ml_kaggle.ipynb` holds only the settings and a `git clone`. Every other cell calls `scripts/kaggle_runner.py` from the cloned repository, so code fixes reach Kaggle with the next run and the notebook never needs re-importing. The notebook:
1. clones this repository;
2. installs the few missing packages;
3. loads the Hugging Face token;
4. imports the dataset;
5. runs the pipeline with `configs/kaggle.yaml`, checkpointing every finished stage to a private Hugging Face repo;
6. writes the submission files, the submission zip and a models bundle to `/kaggle/working`.

### 4.1 One-time: upload the data as a Kaggle Dataset
1. Zip your local `dataset/` folder (the one containing `train/` and `test/`) into `amazon-ml.zip`.
2. On kaggle.com go to **Datasets → New Dataset**, drag in `amazon-ml.zip`, set the title to `amazon-ml`, keep it **Private**, and click **Create**.

Any folder layout inside the zip works. The notebook finds the 7 TSV files wherever they are, and unpacks the archive itself if Kaggle left it zipped.

### 4.2 Create the notebook
1. On kaggle.com go to **Create → New Notebook**, then **File → Import Notebook**, and upload `kaggle/amazon_ml_kaggle.ipynb` (download it from GitHub first).
2. In the right-hand panel, set **Accelerator → GPU T4 x2** and **Internet → On**. Internet requires a phone-verified account.
3. **Add Input → Datasets → Your Datasets →** `amazon-ml`.
4. Add the Hugging Face token for checkpoints:
   - Go to **Add-ons → Secrets → Add Secret**.
   - Set Label to `HF_TOKEN`. Set Value to the part after `HF_TOKEN=` in your local `.env`; the token needs **write** access.
   - Tick the secret's checkbox so it is attached to this notebook.
5. In the first code cell, check `TEAM_NAME` (`neural_nexus`) and `DATASET_SLUG`.

### 4.3 Run
- **Rehearsal first:** set `RUN_SMOKE_FIRST = True` and `RUN_FULL = False`, then **Run All**. It takes about 10–15 minutes and runs the *full-run config* on a 5k-entity sample: the real cross-encoder on both GPUs, plus a real Hugging Face save → restore → skip round trip. It must end with `SMOKE TEST PASSED`.
- **Full run:** set `RUN_SMOKE_FIRST = False` and `RUN_FULL = True`, then **Save Version → Save & Run All (Commit)**. It keeps running after you close the browser. Kaggle stops any session at 12 hours.
- **Results:** open the version's **Output** tab. It contains `matching_results.tsv` (upload this to the portal), `neural_nexus_submission.zip`, `neural_nexus_models.tar.gz`, `reports/` (out-of-fold F0.5 and so on) and `logs/`.

### 4.3.1 Checkpoints and resuming (`ber/checkpoint.py`)
- **What gets saved:** with the token, every finished stage's outputs are pushed together with its completion marker, as one commit, to the **private** dataset repo `<hf-user>/amazon-ml-ber-work`, folder `full/`. Only changed files are uploaded.
- **Mid-stage progress:** the long stages also save their finished pieces while they run, and a later attempt keeps them:
  - `block` saves per country.
  - `prerank` saves its trained models, then its scored chunks every `prerank.sync_every` chunks.
  - `features` saves its shards every `features.sync_every` shards.
  - The cross-encoder saves each trained half, and its scored parts every `ce.sync_minutes`.

  Each piece is written atomically: to a `.tmp` file first, then renamed. A *plan* file records row counts, chunk sizes and the model id, and a piece is reused only when its plan matches. So a crash never leaves a half-written piece that looks finished, and scores from another model or chunk size are never mixed in.
- **Resuming:** if a session stops (12-hour limit, crash, lost connection), commit the notebook again (**Save & Run All**). The first stage call finds an empty work dir, downloads the checkpoint, prints `restored … finished stages: …`, skips those stages and continues inside the interrupted stage from its last saved piece.
- **Out of memory:** if a stage is killed for memory (exit -9/137, -6/134, or 75 from a Python `MemoryError`), the notebook repeats it at once in the same session with lower-memory settings (`LOW_MEMORY_LEVELS` in `scripts/kaggle_runner.py`, two levels). The retry uses fold models one at a time, smaller training samples and fewer workers. The local work dir is intact, so the retry continues where the killed attempt stopped, with no re-download. Separately, fold models train concurrently only when both copies fit in half the free RAM.
- **Tracks (experiments on top of a finished run):** a track config sets `checkpoint.restore_from` (the finished run's
  folder, e.g. `full`, only ever read) and `checkpoint.start_from` (the first stage to re-do).
  - The track restores that run's outputs of the earlier stages only, plus any `keep_stages`. Outputs, resume
    state and completion markers of the re-done stages are never restored, so a re-done stage cannot pick up the
    old results.
  - It writes its own progress to its own folder (`checkpoint.prefix`) and resumes from there after a crash.
  - Ready-made tracks: `configs/track0.yaml` (CPU session: new decision layer + error analysis) and
    `configs/track_a.yaml` (GPU: matchers re-trained from `r1`). Select one with `CONFIG` in the notebook.
  - Two tracks with different prefixes can run at the same time on two accounts.
- **Fresh start:** `FRESH_START = True` deletes the saved progress and starts over. Use it after changing code or settings that affect earlier stages, otherwise stale restored stages are reused.
- **Safety:** the code refuses to upload to a **public** repo, because the checkpoint contains competition-derived data. It reads only `HF_TOKEN` and never prints it.
- **Failures:** an upload failure is logged and retried after the next stage. A missing token just turns checkpointing off.
- **Locally:** set `checkpoint.enabled: true` (`--set checkpoint.enabled=true`); the token is then read from `.env`.

### 4.3.2 Progress bars and the 12-hour budget (`ber/progress.py`)
Every finished stage prints two bars, and the notebook repeats them every 5 minutes while a stage runs, with the latest log line so you can see it is alive:

```
PIPELINE  ███████░░░░░░░░░░░░░░░░░  29%  stage 6/16: prerank | worked 2h54m | remaining ~7h00m (estimate)
SESSION   ██████░░░░░░░░░░░░░░░░░░  25%  of the 12h limit used (2h57m) | 9h02m left | pipeline projected to end at ~9h57m
```
- **How it works:** it reads the completion markers the pipeline already writes, so it cannot drift from what actually ran. Each stage has a rough weight, rescaled by how long finished stages really took.
- **Accuracy:** the first number is only a rough prior (about 5 h). By the time `features` starts, it is typically within a few percent.
- **Overrun warning:** if the projection exceeds the limit, the line ends with `!! may not finish in this session -> commit again to resume from the checkpoint`.
- **Weekly quota:** Kaggle's weekly GPU-hour counter is not visible from code. It is shown in Kaggle's own notebook UI.
- **Elsewhere:** the CLI prints the same PIPELINE bar after every stage, on AWS or locally.

### 4.3.3 2-hop expansion is on
It adds candidates found through a S1's already-confident records. It is applied only if it measurably raises the candidate ceiling, because extra candidates also add noise. On the 5k sample it added +0.00002 to +0.00005 of ceiling, and forcing it on lowered the sample score slightly (0.99740 vs 0.99800), so the gate declines there. On the real data it will apply it only if it finds a real gain. The decision is logged as `2-hop (...): ... gain +X, needs >= 0.00050 -> use=True/False` and saved in `work/models/prerank/expand.json`.

### 4.4 What `configs/kaggle.yaml` changes
Only scale and speed settings change; the code path is the same as on AWS.

| Setting | Kaggle value | Why |
|---|---|---|
| Folds | 4 instead of 5 | Time |
| GBDT | XGBoost on GPU instead of LightGBM on CPU | Kaggle has only 4 CPU cores |
| Random projections / kNN | 192-d instead of 256-d; neighbours 30/3 and 20/3 instead of 50/5 and 30/5 | RAM and time: 40% fewer candidates to score (§4.4.1) |
| Exact-key blocks | At most 50 S1 and 5,000 pairs per block (instead of 200 and 20,000) | Memory. Measured on the full US train data: larger blocks are generic names that add only look-alikes. |
| Model training | Sample of 500k entities | Memory. Every pair is still scored. |
| 2-hop expansion | On, but self-gated: decided on a 250k-entity probe, run on everything only if it raises the candidate ceiling by ≥ 0.0005 | Costs minutes if it does not help; never lowers accuracy when it does not help (§4.3.2) |
| Projection vectors | Saved to the scratch disk (~10 GB of the 1.2 TB), not checkpointed; rebuilt on demand | 2-hop needs them |
| Cross-encoder | `intfloat/multilingual-e5-small` (MIT, 118M parameters) on both T4s, fp16 | Time |
| Work files | On the scratch disk, not the 20 GB `/kaggle/working` | Disk |

### 4.4.1 Accuracy: Kaggle configuration vs the AWS configuration
There **is** a difference. Some settings were reduced to fit the time and memory of a Kaggle session. What was measured and what was not:

**Measured** (18,000 train entities, identical data; "ceiling" = best achievable macro F0.5 of the candidate set, which no later stage can exceed):

| Blocking settings | Ceiling | Pairs/S1 |
|---|---|---|
| AWS defaults (256-d, k 50/5 and 30/5) | 0.99954 | 95 |
| Kaggle before 26 Sep (128-d, k 30/3 and 20/3) | 0.99912 | 57 |
| **Kaggle now (192-d, k 30/3 and 20/3)** | **0.99935** | **57** |
| 256-d, same k (not used: ~25 GB RAM at full scale) | 0.99941 | 57 |
| 192-d + reverse-k back to 5 (not used) | 0.99937 | 70 |

- At full scale the loss is likely larger, because more look-alike records compete for the same neighbours. Measured: −0.00019 at 3k entities and −0.00042 at 18k, so it grows with size. A rough extrapolation says a few tenths of a point at 2.2M entities. That is an upper bound on the score impact.
- **What to watch in the real run:** the `train union: {'union_ceiling': ...}` log line after `block`. If it is below about 0.995, raise the neighbour counts (`blocking.a1/n1.k_fwd`) at the cost of more scoring time.
- The tighter exact-key block caps (50 S1 / 5,000 pairs instead of 200 / 20,000) had no effect at 18k entities. On the real US train data, 3.0% of true pairs sit in name-key blocks that the tighter caps drop. Almost all are still found through the address channels; the rest (missing address) are ambiguous among 50-200 same-name entities anyway.

**Not measured** (they need the real data or GPUs I do not have; each is expected to cost little, none can raise the score):
- **Cross-encoder:** `multilingual-e5-small` (118M, not a reranker) trained on up to 300k positives, instead of `bge-reranker-v2-m3` (568M, a pretrained multilingual reranker) on 1.5M. This is the largest deliberate downgrade of the neural part. It is only one input to round 2.
- **GBDT:** XGBoost on GPU instead of LightGBM; 4 folds instead of 5; training on 500k entities (400k for the pre-ranker) instead of 800k (600k). On the 5k sample the two setups scored within noise of each other.
- **Lookup tables** are mined from 1.5M pairs instead of 3M, so a few rare transliteration entries may fall below the support threshold.

### 4.4.2 Using both GPUs
| Stage | What runs on both GPUs |
|---|---|
| `block`, `expand` (nearest-neighbour search) | The index is copied to each GPU and the query chunks are shared out. The result is identical to a single-GPU run (exact top-k, independent chunks). |
| `prerank`, `r1`, `r2`, `gate` (XGBoost) | The fold models are independent, so two train at once, one per GPU. Saved models are also used for prediction on the GPU. |
| `ce_train`, `ce_infer` (cross-encoder) | The two cross-fitted halves run as separate processes, one per GPU, instead of PyTorch `DataParallel`. |
| Everything else (parsing, features, decision layer) | CPU only (4 cores). |

- **Safety:** each path falls back to one GPU (or the sequential path) if a worker fails, for example on out-of-memory. What was tested: correctness of the search and of concurrent training (identical to single-GPU results), the failure fallbacks, and the worker orchestration, using simulated devices. Not tested: real two-GPU behaviour, which the notebook rehearsal checks.
- **Rehearsal:** it prints `Multi-GPU paths exercised: OK / NOT USED` for the three paths above, from the log.
- **Kill switch:** `EXTRA_OVERRIDES = ["run.max_gpus=1"]` forces single-GPU behaviour in every stage. Per-stage caps: `blocking.knn.max_gpus`, `<stage>.gbdt.max_gpus`, and `ce.parallel_halves: false`.

### 4.5 If the 12-hour limit is tight
Measured on Kaggle (2× T4): `block` takes about 2 h 40 min (exact kNN over ~12.5M train and ~11.7M test records, four searches per country, both GPUs near 100%). The later stages have not been measured at full scale; the estimate is 5–6 h. The notebook prints the session time used after every stage. To shorten a run:
- Set `USE_CROSS_ENCODER = False`. This saves roughly 1–1.5 hours.
- Set `EXTRA_OVERRIDES = ["ingest.train_s1_frac=0.6"]` to train on 60% of the train clusters. The test set is never subsampled.
- Set `EXTRA_OVERRIDES = ["r1.gbdt.rounds=1000", "r2.gbdt.rounds=1000"]`.

With Hugging Face checkpoints on, a session that hits the limit loses at most the last few minutes of the stage that was running: commit again and it resumes (§4.3.1). Without checkpoints, the next run starts over.

---

## 5. Full run on AWS

### 5.1 Machines

| Box | Suggested instance | Used for |
|---|---|---|
| CPU | `r7i.16xlarge` (64 vCPU, 512 GiB) or `r7i.8xlarge` (256 GiB) | every stage except the neural ones (all stages stream, so RAM mostly sets how large the training samples can be) |
| GPU | `g6e.2xlarge` (L40S 48 GB) or `p4d`/`p5` | kNN (much faster on GPU), cross-encoder train/infer, optional bi-encoder |

A single GPU box with enough RAM (for example `g6e.16xlarge`, 512 GiB) can run everything.

- **Request the GPU service quota first.** New accounts often start at zero for G/P instances, and approval can take hours.
- **Without a GPU:**
  - kNN falls back to multithreaded CPU matmul; set `blocking.knn.device: cpu`. This takes hours rather than minutes.
  - Set `ce.enabled: false`.

### 5.2 Commands

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

### 5.3 Outputs

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

## 6. Leaderboard probes

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

## 7. Reproducibility and audit

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
  - For a country without training labels (France), the address administrative level is detected from that split's own addresses (`features/admin.py`), not from a gazetteer.
  - No gazetteers, geocoders or APIs are used.
- **Model licences and size:**
  - LightGBM / XGBoost: trees only.
  - `BAAI/bge-reranker-v2-m3`: Apache-2.0, 568M parameters.
  - `intfloat/multilingual-e5-small` (the cross-encoder of `kaggle.yaml` / `final.yaml`, i.e. the final submission): MIT, 118M parameters.
  - Optional `intfloat/multilingual-e5-base`: MIT, 278M parameters.
  - Total stays well under 8B parameters.

---

## 8. Packaging

```bash
bash scripts/package_submission.sh <team_name>              # -> ../../<team_name>_submission.zip
bash scripts/package_submission.sh <team_name> --with-models   # + ../../<team_name>_models.tar.gz
```

On Kaggle the notebook's last cell runs this for you. It sets `WORK_DIR`, `DATA_DIR` and `OUT_DIR` because those directories live on scratch disk there; `RESULTS_DIR` can also be overridden.

Fill in `docs/Documentation Template.md` before packaging; it is copied into the zip as `Documentation_template.md`.

The local layout (`code/business_entity_resolution/`) is the layout the challenge requires inside the zip, so the same paths work in both places.

---

## 9. Key configuration knobs (`configs/default.yaml`)

| Knob | Meaning |
|---|---|
| `blocking.a1/n1.k_fwd/k_rev` | Neighbours per S1 / per record in the address and name+address channels |
| `blocking.keyblock.*` | Exact-key block caps and the document-frequency band for locality tokens |
| `blocking.dense.enabled` | Optional bi-encoder channel (enable only if the best achievable score needs it) |
| `blocking.save_vectors` | Keep projection vectors on disk (needed by 2-hop expansion and the S1↔S1 diagnostic; rebuilt on demand if missing) |
| `prerank/r1/r2.sample_entities` | S1 entities whose rows train each GBDT; scoring always covers every pair |
| `prerank.chunk_rows`, `features.shard_rows`, `features.join_rows` | Rows processed at a time (lower them if memory is tight) |
| `gbdt_base.backend`, `*.gbdt.device` | `lightgbm` or `xgboost`; XGBoost device `auto`, `cpu` or `cuda` |
| `ingest.train_s1_frac` | Train on a fraction of the train clusters (time valve; test is never subsampled) |
| `expand.probe_entities`, `expand.min_gain` | 2-hop: decide on a sample of N train entities (0 = all); required ceiling gain |
| `run.max_gpus` | Global GPU cap: 0 = use all GPUs (default), 1 = single-GPU everywhere |
| `ce.parallel_halves` | Cross-encoder halves as one process per GPU (needs >= 2 GPUs) |
| `run.cleanup` | Delete the candidate-union files after the pre-ranker has used them |
| `checkpoint.*` | Private Hugging Face checkpointing of the work dir: `enabled`, `repo_id`/`repo_name`, `prefix` (one folder per independent run), `token_env`, `exclude` |
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

## 10. Troubleshooting

- **Out of memory:** on Kaggle the notebook retries a killed stage with lower-memory settings automatically (§4.3.1). If it still fails, lower `prerank.max_train_rows`, `r1/r2.max_train_rows`, `*.sample_entities` and `features.join_rows`. In `block`, also lower `blocking.a1/n1.k_fwd`. Changing `prerank.chunk_rows` or `features.shard_rows` discards that stage's saved partial progress, because they are part of its plan.
- **Kaggle "missing train_source1.tsv …":** the dataset is not attached (**Add Input**), or `DATASET_SLUG` does not match its folder name under `/kaggle/input`.
- **Kaggle `git clone` fails:** switch **Internet** on in the notebook settings (it needs a phone-verified account).
- **Slow kNN:** check that the log line `knn: ... on cuda` appears; tune `blocking.knn.mem_gb` to fit your GPU.
- **Validator warning (a match outside the candidates):** this cannot happen by construction; `outputs` asserts it. If it appears, re-run `expand` and `features` so both files come from the same run.
- **LightGBM missing:** the wrapper falls back to XGBoost automatically (`gbdt_base.backend`).
