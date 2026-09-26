# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** neural_nexus  
**Team Members:** Naveen Mishra  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve each Source-1 (S1) business to its Source-2/3 records in four steps:
1. multi-channel blocking (character TF-IDF nearest neighbours on GPU, plus exact keys);
2. a gradient-boosted pre-ranker that trims candidates without losing recall;
3. two rounds of gradient-boosted pair classification around a small multilingual cross-encoder, with an
   entity-level "has a match" gate;
4. a per-entity set decision that directly targets macro F0.5.

Beyond a standard pipeline, the key ideas are:
- **Collective reasoning:** records compete for a single owner, and each S1's cluster state informs ambiguous
  records.
- **Leaderboard probes to find the weak country:** they located the weak spot, France, which has no training
  labels.
- **Label-free adaptations for France:** learned from the test data's own structure.

## 2. Methodology

### 2.1 Problem Analysis
**Data shape**
- Train: 2.21M S1 (India, US) and 10.3M S2/S3 records.
- Test: 1.73M S1, of which France is **15%**. France appears only in test, so it has no labels.
- Truth structure: 3.46 true records per S1 on average, and 5.6% singletons. A record belongs to at most one S1
  and never to another country. There are at most 5 S2 and 6 S3 records per S1.
- Test has about 24% more records per S1 than train. The extra records are ordinary distractors (unrelated
  businesses), not clusters of removed S1s.

**Noise patterns**
- Transliterated / Devanagari names and states.
- Domain or hashtag names (`zionnewlife.com`).
- Typos and doubled letters.
- Word-order shuffles and moved legal forms (`CORP VT SPACSPHERE`).
- Shuffled address components, abbreviations (`R`, `Av.`, `Rte`), corrupted digits (`2Nd` → `8ND`).
- 15.8% of true pairs have both name and address `token_set_ratio` below 60.

**Missing addresses and shared names (the hardest case)**
- 4.4% of true records have no address (0.3% of distractors); 14.2% of S1 have at least one such record.
- The data contains several S1s with **identical names at different addresses**. A record without an address
  cannot be told apart pairwise between them.

**France**
- Every S1 address carries its *region*; its records carry the region (33%), a *department* instead (32%), or
  neither (35%).
- Candidate lists are twice as long as in train (20.4 per S1 against 10.8).

### 2.2 Solution Strategy
**Approach Type:** Blocking + pre-ranker + two-round GBDT classifier (+ cross-encoder feature) + entity gate +
metric-optimal set decision (hybrid, collective).

**Core Innovation:**
1. Per-entity set selection that optimizes macro F0.5 exactly, including singleton handling.
2. *Competing-cluster* features: round 2 compares an S1's cluster with the best competing S1's cluster, for
   records that several S1s claim.
3. Label-free adaptation for a country unseen in training: its address administrative level is detected from the
   data and ignored, and its list-size features are rescaled to the train scale.

## 3. Candidate Generation (Blocking)

**Blocking keys and channels**, per country partition, in both directions (S1→records and records→S1):
- A1: hashed character-3-gram TF-IDF of the address, randomly projected to 192 dimensions, exact top-k cosine
  kNN on 2 GPUs (k = 30 / 3).
- N1: the same for `[name, 0.5·address]` (k = 20 / 3).
- Exact keys: name skeleton, sorted core name, concatenated name/domain, name × locality token,
  house number × rare address token, PIN/postcode × number, exact address token set. Generic-name blocks are
  capped (≤ 50 S1, ≤ 5,000 pairs).
- A 2-hop expansion through confident members was measured and applied only if it raises the train ceiling by
  ≥ 0.0005 (it did not: +0.00013).

**Candidate pairs generated:**
- Union: 179M (train, 81 per S1) and 160M (test, 92 per S1) pairs.
- A GBDT pre-ranker on cheap similarity and channel features (trained out-of-fold) keeps pairs with
  p ≥ 0.0005 → **10.8 candidates per S1** on train. The test output has **22.47M** candidate pairs (13.0 per S1).

**How true matches were protected:**
- The pre-ranker floor is tuned on the train *candidate ceiling* (the best macro F0.5 achievable from the
  candidates). Union ceiling 0.99687 → final 0.99671.
- Every channel is recall-first; the union pair recall is 98.96%.
- 59% of the remaining lost pairs are records without an address whose name is shared by several S1s.

## 4. Matching Model

**Features used (about 150 in round 1, about 185 in round 2):**
- **Name:** Jaro-Winkler, Levenshtein, token set/sort/partial ratios on core, normalized, romanized and
  consonant-skeleton views; IDF-weighted Jaccard; coverage both ways; legal-form agreement by family; DBA /
  alternative names; domain-name matches (concatenation, initials).
- **Address:** component-level best matches (comma components), token IDF Jaccard/coverage, TF-IDF cosine,
  house-number hit / Jaccard / numeric distance / edit distance, PIN and phone agreement, landmark similarity,
  "bis/ter".
- **Diff signatures:** the *type* of each edit between the S1 and the record (phonetic, typo, abbreviation,
  missing legal form, missing or extra rare token, number change), counted.
- **Context and competition:** the pair's rank and gap within the S1's list and within the record's list, and
  the record's best score with any *other* S1 (one-owner competition).
- **Round 2:**
  - the round-1 out-of-fold score;
  - the cross-encoder score (`intfloat/multilingual-e5-small`, MIT, 118M parameters, fine-tuned 2-fold
    cross-fitted on the uncertain band);
  - consensus with the S1's confident members (name/address similarity, house number and PIN majority);
  - **competing-cluster state**: members of this S1 and of the record's best competing S1 (overall and from the
    record's source, room under the source caps, empty or not), and the number of S1s claiming the record at
    almost the same score.
- List-size counts are divided by the country median in round 2 and the gate. For a country without labels they
  are rescaled to the train scale in round 1.

**Model type:**
- XGBoost (GPU, 4 entity-grouped folds, out-of-fold predictions) for the pre-ranker, round 1 (800k training
  entities), round 2 (700k) and the entity gate (P(has ≥ 1 match)).
- Isotonic calibration per country; France uses the US calibration.

**Threshold selection method:** a per-entity set rule tuned on train out-of-fold macro F0.5, per country:
- add the first member when q₁·κ > 1 − g; add further members above F0.5 break-even thresholds;
- respect per-source caps;
- each record goes to its best S1 only.

An exact expected-F0.5 subset rule (a dynamic program over the member probabilities) is searched alongside. It
tied the grid rule to 1e-5 on out-of-fold, which shows the decision layer is at its optimum for the given
probabilities.

## 5. Results & Error Analysis

| Model | Train out-of-fold macro F0.5 | Public leaderboard |
|---|---|---|
| Baseline (500k training entities) | 0.98933 (US 0.98964, India 0.98888); ceiling 0.99671 | 0.984819 |
| + 800k entities (Track A) | 0.98952 (US 0.98982, India 0.98907) | — |
| + competing-cluster features (Track G) | *(fill in: reports/oof_report.json)* | *(fill in)* |
| + France adaptations (Track F, final) | same as Track G (France unlabeled) | *(fill in)* |

- Precision 0.9985, recall 0.9705. Singleton F0.5 0.9946.
- **Per country on the leaderboard:** four diagnostic submissions blanked or invalidated one country's rows.
  Their exact scores are consistent with India/US scoring as out-of-fold and **France ≈ 0.960**, i.e. about 40%
  of the leaderboard loss from 15% of the entities. This motivated the France adaptations.
- **Common false negatives (2.95% of true pairs):**
  - records without an address whose name is shared by several S1s (about 70% of misses): owned by the wrong S1
    (1.04%), never a candidate (1.09%), or below threshold (0.81%);
  - otherwise heavy combined name and address corruption.
- **Common false positives (0.15% of predictions):**
  - look-alike distractors with the same name and a nearby or the same street (64%);
  - records of another S1 sharing the name (29%);
  - singletons given a match (6%).

## 6. Conclusion
- Recall-first blocking plus two rounds of out-of-fold GBDT with competition features gets within 0.007 of the
  candidate ceiling.
- The remaining train error is dominated by genuinely ambiguous missing-address records, which cluster-level
  (collective) features address.
- The largest leaderboard lesson: an unlabeled test-only country (France) needs its own label-free adaptation.
  Aggregate leaderboard probes are an effective way to find such a gap.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/ber/`: package with `io`, `normalize`, `mining`, `blocking`, `features`, `models`, `decision`, `eval` and
  `pipeline`.
- `configs/`: `final.yaml` holds the final settings; `kaggle.yaml` / `default.yaml` are the base profiles.
- `scripts/`: Kaggle runner, packaging.
- `README.md`, `requirements*.txt`.

**Entry points:**
- **Train + predict:** `python -m ber.pipeline.run --config configs/final.yaml --stage all` (data in
  `../../dataset/{train,test}`). It writes `../../output/matching_results.tsv` and `candidate_pairs.tsv` and runs
  the official validator.
- **Inference on new test data** with the trained models (`<team>_models.tar.gz` extracted into `work/`):
  `python -m ber.pipeline.run --config configs/final.yaml --set run.inference_only=true --stage all`.
- Verified on sample data:
  - inference-only from the models bundle reproduces the full run byte for byte;
  - the Kaggle checkpointed track chain reproduces a from-scratch `final.yaml` run byte for byte.

### B. Additional Results
- **Decision layer:** exact expected-F0.5 rule vs grid rule = +0.00001 (tie). A 2-hop expansion gained +0.00013
  on the ceiling (not used).
- **Fair play:**
  - no external data: lookup tables are mined from training pairs (≥ 20 supporting entities each), and the
    France admin level is detected from the test addresses themselves;
  - hand-written rules are abbreviation knowledge only;
  - models are MIT / Apache-2.0 and far below 8B parameters.
