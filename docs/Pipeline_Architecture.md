# Business Entity Resolution: Pipeline Architecture (v2)

Amazon ML Challenge 2026 · Team working document · Written 25 Sep 2026

This document is the build plan for our submission. It explains what the data looks like, how the metric works, the full pipeline stage by stage, how we validate, and how we get to the highest private-leaderboard F0.5 within the challenge window. It also records which external review suggestions we adopted or rejected, and why.

---

## 0. Summary

**The problem as the data shows it.** Each Source 2 / Source 3 record belongs to **at most one** Source 1 entity. Matches never cross countries. Source 1 is clean and all the noise is on the S2/S3 side. Groups are small: at most 5 S2 records and 6 S3 records per S1 entity. So the task is to **decide, for every noisy record, which clean S1 entity owns it, or none**, and then pick for each S1 the set of records that maximises per-entity F0.5.

**The pipeline:**

| Stage | What it does | Output |
|---|---|---|
| 0. Multi-view normalisation | Parsing rules chosen by country with a generic fallback. Keeps raw, cleaned, Latin-script and consonant-skeleton versions of every field. The clean S1 address guides parsing of the noisy record. | Parsed tables (Parquet) |
| 1. Candidate generation | Lexical blocking from both directions, in several channels, split by country. Then a learned pre-ranker with a score floor, then 2-hop expansion through record↔record links. Judged by the **best achievable macro-F0.5 of the candidate set**. | `candidate_pairs.tsv` |
| 2. Pair scoring | Round-1 LightGBM on similarity, number, **"what changed"** and competition features. A cross-encoder on the uncertain middle. Round-2 LightGBM with group-consensus features, trained on out-of-fold round-1 scores. | Calibrated pair probabilities |
| 3. Decision layer | Each record goes to its best S1 only. An **entity gate** estimates P(S1 has any match). Then top-1, plus extra members above size-dependent thresholds, within per-source caps. Thresholds are tuned per country. | `matching_results.tsv` |

**Validation.** Every training S1 gets out-of-fold scores from 5-fold GroupKFold. The decision layer is tuned on the full training set. A stress test reproduces the test set's higher density of unmatched records.

**France** (15% of test, no labels). A French parsing profile, features that don't depend on country, optional monotone constraints, and paired leaderboard probes that isolate France's score.

**Models.** About 1B parameters in total, all MIT or Apache-2.0. That is far under the 8B cap even on the strictest reading.

---

## 1. Constraints

| Constraint | Detail | Consequence |
|---|---|---|
| Window | 25 Sep 00:00 IST → **27 Sep 23:59 IST** (Guidelines.pdf) | Staged plan. A valid submission on day 1, and every later component must be strictly additive. |
| Submissions | **5 per day**, 15 in total | Submissions are scarce. Use them to answer questions validation cannot (§12). |
| Scoring | Public LB is a subset of test; the final ranking uses the private LB | Optimise out-of-fold validation. Use the LB only for things validation cannot see: France and shift. |
| Models | MIT/Apache-2.0, "up to 8B parameters" | We read the cap as the **pipeline total** (strict reading). Budget ≈1B (§13). |
| External data | No lookup APIs, geocoders, registries or internet data | All lookup tables are **mined from training pairs** or are small hand-written abbreviation rules. No downloaded gazetteers or postcode lists. |
| `candidate_pairs.tsv` | Must be the **last** candidate set the matching model runs on; matches ⊆ candidates | All candidate expansion (including 2-hop) happens **before** this file is written. The decision layer only removes. |
| Audit | Top teams' zips are reproduced | Seeded, checkpointed, one command per stage, and an inference-only path from saved weights (§16). |
| Compute | AWS cloud, effectively unlimited for the window; the local laptop (16 GB / 4 GB VRAM) is for small-sample development only | Full-scale runs go on cloud instances (§14). |

---

## 2. Verified data facts

All numbers below were computed directly from the provided TSVs on 25 Sep. Counts come from full-file passes unless marked *sample*. The *sample* rows used every 50th ground-truth row: 44,136 S1 entities and 153,241 matched pairs. They will be recomputed by `src/.../eda/profile.py` so the numbers are reproducible.

### 2.1 Sizes and density

| | S1 | S2 | S3 | S2+S3 per S1 |
|---|---|---|---|---|
| Train total | 2,206,821 | 5,034,616 | 5,285,603 | **4.68** |
| Train US | 1,323,633 | 3,016,817 | 3,170,056 | 4.67 |
| Train India | 883,188 | 2,017,799 | 2,115,547 | 4.68 |
| Test total | 1,732,544 | 4,887,273 | 5,082,316 | **5.75** |
| Test US | 663,106 | 1,871,330 | 1,945,701 | 5.76 |
| Test India | 809,986 | 2,312,565 | 2,405,000 | 5.82 |
| Test France | 259,452 | 703,378 | 731,615 | 5.53 |

### 2.2 Structure of the ground truth

| Fact | Value | Consequence |
|---|---|---|
| Records matched to >1 S1 | **0 of 7,638,365** | Each record has at most one owner, so take the best S1 per record (§9.1). No bipartite or Hungarian matching is needed, because an S1 can own many records. |
| Unmatched S2/S3 records in train | 2,681,854 (**26.0%**) | Plausible-looking records that match nothing. |
| Matched pairs that cross countries | **0 of 153,241** (*sample*) | Split everything by country and treat a country mismatch as a hard filter. |
| Singletons (S1 with no match) | 123,247 (**5.58%**). US 5.58%, India 5.59% | Singletons are a small share of entities. The gate matters most where top-1 is weak (§3). |
| Group size (S2+S3), 0→11 | 0:123,247 · 1:119,157 · 2:375,212 · 3:530,841 · 4:484,115 · 5:321,957 · 6:164,868 · 7:63,968 · 8:18,680 · 9:4,205 · 10:534 · 11:37 | Mean 3.46, identical for US and India. The largest group has 11 records. |
| S2 records per S1 | 0–**5** (max) | Hard cap: never predict more than 5 S2 records for one S1. |
| S3 records per S1 | 0–**6** (max; 6 occurs for only 2,816 S1s) | Hard cap: at most 6 S3 records. |
| S1 entities sharing a normalised name | 870,893 (**39.5%**), e.g. "Primary Care Group" ×253 | The name alone cannot decide; locality and address must. |
| S1 entities sharing an exact normalised address | 120,493 (**5.46%**), up to 14 per address | Record 'same address, different business' traps. An address-ambiguity feature is essential. |
| Near-duplicate non-matches exist | "Classic Equity Partners **Group**, **6885** Catalpa Bluff Ln" is **not** a match for "Classic Equity Partners **Inc**, **6876** Catalpa Bluff Lane" | Number and "what changed" features (§8.2). |
| True matches whose name shares no ASCII token with the S1 name | **14.34%** (*sample*) | These are cross-script, domain-style and trade names. Address channels are required. |
| True matches with an empty or null address | 4.44% (*sample*) | Name-only channels are required. |
| True matches with neither a shared name token nor an address | **0.004%** (*sample*) | Union of name and address channels → near-perfect recall ceiling is achievable lexically. |
| Non-empty matched addresses sharing no token with S1 | 0.02% (*sample*) | Address token overlap is almost universal for true matches. |

### 2.3 Where the noise is

| File | Indic-script name | Domain-style name | Empty address | null/N/A token in address |
|---|---|---|---|---|
| train S1 | 0.0% | 0.0% | 0.0% | 0.0% |
| train S2 | 9.4% | 4.0% | 3.4% | 3.5% |
| train S3 | 5.3% | 4.0% | 3.3% | 3.3% |
| test S1 | 0.0% | 0.0% | 0.0% | 0.0% |
| test S2 | 11.2% | 3.2% | 2.6% | 2.9% |
| test S3 | 6.3% | 3.2% | 2.7% | 2.8% |

**S1 has none of these heavy noise types, in train or test.** It does carry light noise: junk prefixes (`<< Team Ecole`) and reordered address components (`IA, Iowa City, 1064 Newton Rd`). Almost all the noise is on the record side. So we:
- Treat S1 as the reference: parse it reliably, then align the noisy record to it (§6.4).
- Use **asymmetric** features: how much of S1 the record covers, and how much of the record S1 explains.

The Indic share per India record is the same in train and test (≈23–24%). The only change is that India makes up a bigger share of test.

### 2.4 Observed noise types (from sampled clusters)

- **Names:**
  - Typos: *Wilblims, Sterlnig, Tetlecommunication*.
  - Injected accents: *Téchnology, Àmicale*.
  - Case changes.
  - Junk prefixes: `--`, `<<`, `#`, `M/s`, `Shri`, `Smt`.
  - Junk suffixes: `#8180`, *Center, Company, Service*.
  - Brackets: `[Tele]`, `[EURL]`.
  - The legal form moved or changed: *LLC Moncada…, Pvt. EFS … Ltd., S.A.S, प्रा. लि.*
  - Words reordered.
  - Domain-style names: `maurewilliamscolombier.com`, `rcprivate.com` (built from initials), `www.indiaaps.com` appended after `|`.
  - Indic-script transliteration of the full name or part of it.
  - Trade names that share nothing with the S1 name (*Evowex, Dréxkor*).
  - Digit-for-letter swaps (found while building the code): *N0se, C0mmunity, Techn0logies, 5ervices*, plus `lnc` for `Inc` and `lndia` for `India`. Mixed letter+digit tokens are repaired (`0→o`, `5→s`, …); tokens like `3M` are left alone.
  - Alias phrases: *d/b/a, t/a, aka, fka, formerly known as*. The name is split into alternatives.
- **Addresses:**
  - Parts reordered.
  - Case changes.
  - Abbreviations: *St, Rd, Ave, R., AV*.
  - State written as a code, a full name or in native script (*TN / Tamil Nadu / தமிழ்நாடு*).
  - City aliases (*Calcutta / Kolkata / Kolkatta*).
  - An extra or wrong district, township or département (*Howrah, Hugli; Ticonderoga Townshiip; Nord / Gironde* in place of the region).
  - Changed house numbers (*45th → 45ND, AF-684 → AF-0684, 9 → 17* inside true matches).
  - Prefixes such as `DOOR NO`, `H.no #`, `NO.`.
  - Phone numbers inside the address.
  - Landmarks (*Near Mother India Public School*).
  - Missing parts, and `null` / `N/A` tokens.
- **France (test only):**
  - *R. / R / AV* for Rue / Avenue.
  - `9 B` = 9 bis.
  - `NO. 5 ALLÉE…`.
  - `(41) Rue…`.
  - Département in place of region.
  - *Cie*.
  - Dotted legal forms (*S.A.S*, *S.A.*).
  - The same kinds of typos and accent injection.

**Key observation:** the test's extra density is **uniform across countries** (5.76 / 5.82 / 5.53 vs 4.68). If groups in test have the same size as in train (mean 3.46), test has **about 2.3 unmatched records per S1 vs 1.2 in train**. That is the same density we get by randomly deleting **about 19% of the training S1 entities** and keeping their records as unmatched. We treat this as a hypothesis to confirm (§10.4), not a fact.

---

## 3. Metric analysis

F0.5 is computed **per S1 entity** and then macro-averaged. For one entity with TP, FP, FN:

```
F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP)
empty prediction:     1.0 if the entity is a singleton, else 0.0
non-empty prediction: 0.0 if the entity is a singleton
```

### 3.1 The first member is a different decision from the rest

A false top-1 on an entity that **has** matches scores 0, and so does predicting empty. **Relative to empty, a wrong top-1 costs something only when the entity is a singleton.** So the rule for the first member is:

```
predict {top-1}  iff  P(top-1 is correct) · E[F | top-1 correct]  >  P(entity is a singleton)
```

- E[F | top-1 correct, predicting only it] = 1.25 / (0.25·m + 1) for a true group of size m: 1.00, 0.83, 0.71 and 0.63 for m = 1–4.
- The singleton prior is only 5.6%, so for entities that clearly have matches the first-member cutoff can sit **well below 0.5**. A single global cutoff of about 0.7 (v1's advice) would wrongly empty every entity whose best candidate scores 0.5–0.7.
- The entity gate (§9.2) estimates P(singleton) directly. It does not use the product ∏(1−pᵢ), because candidates for look-alike names are strongly dependent and that product badly underestimates P(singleton).

### 3.2 Extra members are precision-weighted

Suppose k members are already selected (assume they are correct) and a new candidate has probability p. Adding it helps in expectation only if p > t(k):

| k already selected | 1 | 2 | 3 | 4 | 5 | 6 | → ∞ |
|---|---|---|---|---|---|---|---|
| t(k) | 0.727 | 0.759 | 0.771 | 0.778 | 0.782 | 0.785 | 0.800 |

The derivation is in Appendix B. These values start the grid search; the decision layer tunes an offset δ per country (§9.4).

### 3.3 Where the points are

- **Group sizes 2–5 are 78% of entities.** A missed member costs little: 3 of 4 correct scores 0.94. A false extra member costs a lot: 4 correct + 1 wrong scores 0.83.
- **Singletons (5.6%)** are worth a full point each, but only if nothing is predicted.
- **Entities whose true matches are missing from the candidate set** score 0 whatever we do. So blocking is judged by the best achievable score, not by pair recall (§7.6).
- **Assigning a record to the wrong S1 costs twice:** it adds a false positive to one entity and a missed match to another. The one-owner rule (§9.1) guards against both.

---

## 4. Review of external critiques (v1 → v2)

Four LLM reviews of the v1 plan were read. Response 4 arrived truncated after its first paragraph. Each suggestion was checked against the data and the metric; only robust, helpful ones were adopted.

| # | Suggestion | From | Verdict | Reason |
|---|---|---|---|---|
| 1 | The first-member decision is P(top-1 correct) vs P(singleton), not a 0.7 cutoff | R1, R3 | **Adopted** | Correct by derivation (§3.1). v1 was wrong here. |
| 2 | Add an entity gate P(≥1 match) and drop the independence-based expected-F search | R1, R2, R3 | **Adopted** | Look-alike candidates are dependent; a learned gate is robust (§9.2). |
| 3 | Replace the hard "gap to runner-up" rule with competition-aware probabilities | R1 | **Adopted** | A hard gap rule drops a record from **both** S1s. Gap features go into round 2 instead (§8.5). |
| 4 | Use GroupKFold out-of-fold scores for all training S1s instead of a 20% holdout | R1 | **Adopted** | A holdout biases the one-owner competition either way; out-of-fold scores give a fair, full-scale tuning set (§10). |
| 5 | Mine lookup tables inside the folds; train round 2 on out-of-fold round-1 scores | R1 | **Adopted** (tables: via a ≥ 20-entity support rule, §6.6) | Prevents leaks, e.g. a transliteration dictionary that has memorised training names. |
| 6 | Judge blocking by the best achievable macro-F0.5, use a score floor instead of a fixed top-25 | R1, R2, R3 | **Adopted** | Pair recall over-weights big groups; K should be measured, not guessed (§7.6). |
| 7 | Check the unmatched-record hypothesis before simulating it | R1 | **Adopted** | Density is up uniformly across countries (§2.1); confirm with score distributions (§10.4). |
| 8 | All-empty submission to learn the public singleton rate | R1 | **Adopted** | Costs one slot on day 1, when slots are spare anyway. |
| 9 | S1↔S1 diagnostic (S1 is deduplicated, so S1↔S1 pairs are true negatives) | R1 | **Adopted, diagnostic only** | Compares false-positive tendency across countries including France. Not used for training on test. |
| 10 | 2-hop / transitive candidates added **before** writing `candidate_pairs.tsv` | R1, R2, R3 | **Adopted** | Recovers trade-name and empty-address members while keeping matches ⊆ candidates (§7.5). |
| 11 | Split oversized name blocks by locality instead of dropping them | R1 | **Adopted** | 39.5% of S1 names are shared (§7.3). |
| 12 | Build lexical blocking first; add dense search only if the best-achievable-score analysis shows a gap | R1, R2 | **Adopted** | Data supports it: 0.004% of true pairs share neither a name nor an address token (§7.7). |
| 13 | Country-dispatched parsing (e.g. "St" = Saint in France); keep country out of the model features | R1 | **Adopted** | Follows the organisers' tip about country-specific address patterns, without a closed-set feature that breaks on France. |
| 14 | Keep raw + normalised + transliterated views; compute features on each | R1, R3 | **Adopted** | Transliteration loses information; more views give the model more evidence. |
| 15 | Rule-based romaniser + consonant skeleton instead of a custom character-level seq2seq model | R1 | **Adopted** | Faster; handles Tamil k/g and t/d, and Hindi schwa deletion. |
| 16 | Phone, landmark and Indian municipal numbering as separate parsed fields; DBA split | R1 | **Adopted** | Keeps the number features clean (§6.4). |
| 17 | Address as components, not only a bag of words | R3 | **Adopted** | Order is shuffled *between* components, not within them; we compare at component level plus a bag-of-tokens view (§6.4). |
| 18 | **"What changed" features**: typed differences between the two records | R1 | **Adopted, high priority** | The way to separate planted traps from noisy true matches on synthetic data (§8.2). |
| 19 | Source (S2 vs S3) as a feature; check per-source caps | R1, R3 | **Adopted** | Verified caps: S2 ≤ 5, S3 ≤ 6 (§2.2). |
| 20 | Group-agreement features (house number / PIN vs confident members) | R1, R3 | **Adopted** | Round 2 (§8.5). |
| 21 | Cross-encoder only on the uncertain band, fed by the cheap model | R1, R2, R3 | **Adopted** | Places compute where it helps most. |
| 22 | Hard-negative taxonomy for the neural models | R3 | **Adopted** | §8.4. |
| 23 | Report singleton / non-singleton / per-country / per-group-size metrics | R3 | **Adopted** | §10.3. |
| 24 | Strict 8B reading; drop the Qwen-7B LoRA | R1, R2 | **Adopted** | A disqualification risk is not worth it. Budget ≈1B. |
| 25 | Subsample training by entity, not by pair | R1 | **Adopted** | Keeps candidate lists whole for ranking features and the gate. |
| 26 | Reproducible inference-only path from saved weights | R2 | **Adopted** | §16. |
| 27 | Hungarian / bipartite matching | R2 | **Rejected** | The relationship is many-to-one; taking the best S1 per record is already optimal. Hungarian forces one-to-one and would destroy recall. |
| 28 | Conservative global cutoff of 0.65–0.75 | R2 | **Rejected** | The same metric error as v1 (§3.1). |
| 29 | "25–40% singletons", "clusters of 30–50+" | R2 | **Rejected** | Data says 5.6% and at most 11 (§2.2). |
| 30 | Cross-encoder at 200–350 pairs/s means 40 GPU-hours | R2 | **Rejected (numbers)** | Short pair texts (≈100 tokens) run in the thousands of pairs/s on one modern GPU. The band-limiting conclusion was adopted anyway (#21). |
| 31 | Avoid leaderboard probing entirely | R2 | **Rejected** | Paired probes that change only France rows have low variance on a large public subset, and France has no other signal. We keep probes few and coarse (§12). |
| 32 | Only about 60 features, no cross-encoder | R2 | **Rejected** | Too thin to win on planted near-duplicates. |
| 33 | "The EDA statistics are hallucinated" | R3, R4 | **Rejected** | The reviewers only saw the problem statement. Every statistic was computed from the TSVs (§2, Appendix C). |
| 34 | Use `country_same` / `country_pair` / country categorical as model features | R3 | **Rejected** | `country_same` is always 1 after splitting by country. A country categorical cannot extrapolate to unseen France. Country is used for parsing and threshold sets only. |
| 35 | Separate S1-S2 and S1-S3 models | R3 | **Deferred to ablation** | One model with a source feature uses the data better; trees learn the interactions. |
| 36 | Entity-balanced sample weights in the pair model | R3 | **Deferred to ablation** | Unweighted training keeps probabilities calibrated. Macro alignment is done in the decision layer. |
| 37 | Pseudo-labelling the unlabelled French test records | v1 | **Not used unless organisers confirm in writing** | A grey area under "only the provided training data". |

---

## 5. Architecture overview

```
                            dataset/{train,test}/*.tsv
                                        │
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ STAGE 0  Ingest → split by country → multi-view normalisation                  │
│   views: raw | norm | latin (romanised) | skel (consonant skeleton)            │
│   name:  core · legal_form · affixes · DBA alternatives · domain parts         │
│   addr:  components · house/unit number sequence · street · locality ·         │
│          city · state · PIN/postcode · landmark · phone · missing flags        │
│   S1 parsed as the reference; record components aligned to S1's components     │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ STAGE 1  Candidate generation (per country, S1→record AND record→S1)           │
│   A1 address char-3gram TF-IDF    A2 house-no × rare street / PIN keys         │
│   N1 name(+locality) TF-IDF       N2 name keys (skeleton, sorted, domain)      │
│   [D1 dense bi-encoder, only if the best-achievable-score gap justifies it]    │
│   union → pre-ranker (LightGBM, cheap features) → score floor                  │
│   → 2-hop expansion via record↔record kNN from confident members               │
│   ══> candidate_pairs.tsv   (tracked: best achievable macro-F0.5, cands/S1)    │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ STAGE 2  Pair scoring                                                          │
│   R1 LightGBM: similarity × views · numbers · "what changed" · competition ·   │
│      ambiguity · source · channel flags                                        │
│   CE cross-encoder on the uncertain band (0.02 < p < 0.98, plus top-3 per S1)  │
│   R2 LightGBM (stacked, trained on out-of-fold R1 + CE):                       │
│      group-consensus + competitor features                                     │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ STAGE 3  Decision layer                                                        │
│   ownership: each record → its best S1 (others zeroed)                         │
│   entity gate: P(S1 has ≥1 match) from its candidate-list profile              │
│   select: empty | top-1 | top-1 + members with p > t(k)+δ                       │
│   caps: S2 ≤ 5, S3 ≤ 6    thresholds per country; France starts from US        │
│   ══> matching_results.tsv  (⊆ candidate_pairs.tsv, validated)                 │
└───────────────────────────────────────────────────────────────────────────────┘
Validation: 5-fold GroupKFold out-of-fold over all 2.2M training S1s ·
            best-achievable-score tracking · unmatched-density stress test ·
            leave-one-country-out · S1↔S1 diagnostic · leaderboard probes
```

---

## 6. Stage 0: Ingest and multi-view normalisation

### 6.1 Ingest

- Read with Polars (`separator="\t"`, `quote_char=None`, all columns as strings) and write to Parquet once.
- Map IDs to `int64` codes: a source code plus the numeric part. Keep the original ID strings in a lookup table for output.
- **Split everything by `country`**, using the label set present in each file; nothing hard-codes {US, India}. Verified: no matched pair crosses countries (0 / 153,241). A country missing from train (France) is processed like any other.
- Shard each partition by `hash(S1 id) mod N` so stages can run in parallel and resume after failure.

### 6.2 Views

Every name and address field keeps four views, and features are computed on each:

| View | Construction |
|---|---|
| `raw` | Original string |
| `norm` | NFKC, casefold, accents folded; `& → and` (US/IN) or `& → et` (FR); punctuation → space; junk affixes stripped; `null / n/a / none / -` → missing; whitespace collapsed |
| `latin` | `norm` after romanising any Indic-script span with a permissive rule-based romaniser (`indic-transliteration` MIT, or `anyascii` ISC). The mined token dictionary (§6.6) is applied first for frequent tokens. |
| `skel` | Consonant skeleton of `latin`: drop vowels after the first letter; merge voiced/unvoiced pairs (k/g, t/d, p/b, c/j, s/z); merge aspirates (kh→k, th→t, bh→b, ph→f, sh→s); collapse doubled letters. `प्राइवेट` and `private` → `prvt`; `டிரேடர்ஸ்` and `traders` → `trtrs`. |

### 6.3 Name parsing

For each name, extract:

- **Legal form.** Uses a per-country profile plus a generic list, and is matched in any position and with dots removed.
  - US: Inc, Corp, LLC, LLP, LP, Ltd, Co, PLLC, PC.
  - India: Pvt Ltd, Private Limited, Ltd, LLP, OPC, and their Indic forms (`प्रा. लि.`, `प्राइवेट लिमिटेड`, `எல்எல்பி`, …) mined from the training pairs.
  - France: SARL, SAS, SASU, EURL, SA, SCI, SNC, SCOP, EI, EIRL, including dotted forms.
  - Output: a canonical legal-form class plus a "had legal form" flag.
- **Affixes.** Honorifics and prefixes (`M/s`, `Shri`, `Smt`, `The`), store numbers (`#8180`), and trailing generic words mined from the training pairs (*Center, Company, Group, Service, …*). These are stripped from the core and kept as a separate token set.
- **DBA alternatives.** Split on `d/b/a`, `dba`, `t/a`, `aka`, `|`, and on bracketed content. The name becomes a list of alternatives, and features take the best-matching alternative.
- **Domain parts.** Detect `www.`, `.com / .in / .fr / .net / .org`, and names with no spaces. Strip the TLD and `www`. Pairwise features check whether the S1 core tokens joined together equal the domain body, a prefix of it, or its initials (`rcprivate` = `r`+`c`+`private`). This needs no vocabulary, so it avoids segmentation errors.
- **Core name.** What remains after the steps above, in all four views, plus a sorted-token version.

### 6.4 Address parsing, guided by the clean S1 address

Addresses are **shuffled between comma-separated components, not within them**. So we parse at the component level and never rely on a single brittle parser of noisy text.

1. **Parse S1 with the country profile.** Its formats are regular, for example `number street, [unit], city, ST` (US) and `…, locality, city, state` (India). The result is typed components: house or door number sequence, unit, street, locality, city, district, state, PIN/postcode, landmark, phone.
2. **Extract typed fields from the record independently:**
   - **Number sequences.** `D.No 45-2-17`, `12/3`, `AF-0684`, `6(29)` and `(41)` become sequences of parts with leading zeros removed. Ordinal suffixes are normalised (`45th / 45nd → 45`) and `bis/ter/B` suffixes are recorded (`9 B` = `9 bis`).
   - **PIN / postcode.** India: 6 digits. US and France: 5 digits.
   - **Phone.** Runs of 7 or more digits, or text after `Ph.` / `Mob`. Phone numbers go into their own field and never into the house-number set.
   - **Landmark.** Text after `near / nr / opp / behind / beside / next to / adjacent`. Digits inside a landmark stay in the landmark field.
   - **Prefix tokens.** `door no`, `h.no`, `no.`, `#`, `plot`, `flat`, `unit`, `apt`, `ste` are dropped, but the number that follows them is kept.
3. **Align record components to S1 components.** Each record component is matched to the most similar S1 component using a maximum-weight assignment over component similarity (≤ 8×8, numba). This yields per-type similarities (street↔street, city↔city, …), "extra" record components (district/département/township injections, landmarks) and "missing" S1 components. Record fields are typed by what they aligned to in the clean S1, so no standalone parser for the noisy record is needed.
4. **Keep a bag-of-tokens view** as an extra signal, never the only one.

State/region equivalences (`TX↔Texas`, `TN↔Tamil Nadu↔தமிழ்நாடு`, `MH↔Maharashtra`) and city aliases (`Calcutta↔Kolkata`, `Bangalore↔Bengaluru`) come from the mined tables (§6.6). A département in place of a region, or an injected district, is treated as a **soft** mismatch because it is a noise pattern, not a trap.

### 6.5 Country profiles

`profiles/{us,india,france,generic}.py` hold only **parsing and normalisation rules**: abbreviation expansion, legal forms, number formats, postcode length and the meaning of `&`. Unknown countries fall back to `generic`, which has no abbreviation expansion beyond universal punctuation rules.

| Rule | US | India | France |
|---|---|---|---|
| `St` | Street | Street | **Saint** (`Ste` = Sainte) |
| `Dr` | Drive (in address), Doctor (in name) | Doctor / Drive | Docteur |
| Street types | St, Rd, Ave, Blvd, Dr, Ln, Ct, Pl, Ter, Trl, Hwy, Pkwy, Cir | Rd, Marg, Nagar, Colony, Cross, Main, Layout, Sector | R./R = Rue, Av/AV = Avenue, Bd = Boulevard, Pl = Place, Rte = Route, Ch = Chemin, Imp = Impasse, All = Allée, Fbg = Faubourg, Sq = Square |
| Number suffix | 1/2, Unit, Apt, Ste | `-`, `/`, `()` multi-part | bis / ter / quater, B |
| Postcode | 5 digits | 6-digit PIN | 5 digits |
| Stopwords in names | of, the | – | de, du, des, la, le, les, l', et |

These rules are hand-written domain knowledge about abbreviations, not an external dataset. No gazetteers, postcode lists or city lists are downloaded.

### 6.6 Lookup tables mined from training pairs

From matched training pairs, align tokens (edit-distance and skeleton matching) and count the substitutions:

- Abbreviation pairs (`st↔street`, `rd↔road`).
- State and city equivalences.
- Indic-token → Latin-token dictionary (`प्राइवेट→private`, `லிமிடெட்→limited`).
- Affixes that noise injects (words added in positive pairs).

Only keep an entry that is supported by **≥ N distinct S1 entities** (N = 20) and that accounts for ≥ 60% of the source token's mappings. This keeps generic mappings and drops memorised entity names.

**As implemented:** one global mining run (`ber/mining/tables.py`, 3M sampled pairs, three passes: A token/abbreviation, B component equivalences, C affixes). This is used for both out-of-fold validation and test. It is leak-safe by construction: an entry supported by ≥ 20 distinct entities would also be found on 4/5 of the data (≥ 16 entities), so fold-internal mining would give essentially the same tables at five times the cost.

Abbreviation mining only accepts true abbreviations: 2–5 letters, at least 2 letters shorter than the long form, and a subsequence of it. This keeps doubled-letter typos (`city/ccity`, `aenue/avenue`) out of the table.

---

## 7. Stage 1: Candidate generation

Everything runs per country partition and in **both directions**. Each record retrieves its top S1s, because a record has at most one owner and reverse retrieval makes sure the true owner is present for the ownership step.

### 7.1 Channels

| Channel | Index | Query | k (S1→rec / rec→S1) | Mainly catches |
|---|---|---|---|---|
| **A1** Address TF-IDF | char 3-gram TF-IDF on the `latin` address (numbers kept) | same | 50 / 5 | Cross-script names, trade names, domain names (address intact) |
| **A2** Address keys | inverted index on (number-sequence head + rarest street token), (PIN + number head), (street skeleton + city) | exact | all in block | Heavy typos in the address, reordered components |
| **N1** Name + locality TF-IDF | char 3–4-gram TF-IDF on `latin` core name + city/state tokens | same | 30 / 5 | Records with an empty address; name typos |
| **N2** Name keys | skeleton of the first two core tokens + state; sorted core tokens; joined core (domain); initials + city | exact | all in block | Transliteration, domain names, reordering |
| **D1** Dense (conditional) | fine-tuned small multilingual bi-encoder, one vector for "name \| address" | exact GPU kNN | 30 / 5 | Only if the §7.6 analysis shows residual misses |

**As implemented (`ber/blocking/`):**
- Char-3-gram counts are hashed (2¹⁸ features, `char_wb`), weighted by TF-IDF (sublinear tf, IDF per country partition) and **random-projected to 256-d float16**.
- Random projection preserves cosine in expectation and keeps TF-IDF's emphasis on rare n-grams, which SVD would lose.
- Top-k is an **exact** blocked matmul + `topk` in PyTorch: on the GPU when available, otherwise multithreaded on the CPU.
- N1 uses the concatenation [name, 0.5·address], so records with no address still get a name-only match.
- Exact sparse TF-IDF cosines for candidate pairs are recomputed from the saved IDF as round-1 features.

### 7.2 Why these channels

- 85.7% of true pairs share a name token (*sample*); N1 and N2 cover these.
- 95.6% have a non-empty address, and 99.98% of those share an address token; A1 and A2 cover these.
- Only **0.004%** have neither.

So a lexical union can reach a near-perfect best-achievable score. The real difficulty is **ranking within oversized blocks**, not coverage.

### 7.3 Oversized blocks

When a key's block has more than B_max (≈200) members, it is subdivided by adding a locality token (city → PIN → street skeleton). It is not dropped, because dropping removes exactly the generic-name entities. If a block is still too large after subdivision, that key is skipped for that block; the TF-IDF channels still cover it. Every block size is logged and becomes an ambiguity feature.

### 7.4 Pre-ranker and score floor

- **Model.** LightGBM on about 30 cheap features: channel flags and ranks, TF-IDF cosines, token overlaps, number-sequence agreement, legal-form agreement, and name/address block sizes.
- **Training.** Out-of-fold, same folds as §10.
- **What is kept:**
  - Every pair with `p_pre ≥ floor`.
  - For every record, its top-2 S1s by `p_pre` if above `floor_low`, so the ownership competition stays intact.
  - A safety cap of 60 per S1. This should almost never bind, since the largest group has 11 records.
- **Choosing the floor.** The largest value whose best-achievable score is within **0.0002** of the unfiltered union. Blocking losses cannot be recovered later, and extra candidates are cheap.

### 7.5 2-hop expansion (before writing the candidate file)

1. **Record↔record kNN** per country: TF-IDF on `latin` name + address over S2 ∪ S3, top-5 neighbours with cosine ≥ c_min.
2. **Confident members:** for each S1 e, `M(e) = {r : p_pre(e,r) ≥ 0.9}`.
3. **Add neighbours:** the kNN neighbours of M(e) that are not already candidates are added to C(e) with a `hop2` flag. At most 10 are added per S1.

This recovers members that share little with S1 but a lot with its other copies (e.g. a trade name at the same address as confident members, or an empty-address record with the same name as a confident member). **This is only run if it measurably raises the best-achievable score.**

### 7.6 Output and metrics

`candidate_pairs.tsv` is the final C(e) after the floor and the 2-hop step: one row per S1, including empty rows, no duplicates, S2/S3 IDs only. We track:

- **Best achievable macro-F0.5** (primary): for each entity, the F0.5 of predicting exactly the true matches that are in the candidate set, averaged over all S1s (Appendix A).
- Pair recall; candidates per S1 (mean, p99); reduction ratio; how many true pairs each channel finds alone.
- The above broken down by country, source, group size and noise type (Indic, domain, empty address).

### 7.7 Dense channel: only if needed

Train it only if the misses in §7.6 are dominated by cross-script or typo-in-both cases that lexical channels cannot fix.

- **Model:** `intfloat/multilingual-e5-base` (278M, MIT) or `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0).
- **Training:** loss = in-batch negatives (MultipleNegativesRanking) + one mined hard negative per positive. Batches are built within a country so in-batch negatives are hard.
- **Cross-fitting:** two folds (each half embeds the other), so validation retrieval is not optimistic.
- **Always useful anyway:** the embedding cosine is also a cheap feature for Stage 2.

---

## 8. Stage 2: Pair scoring

### 8.1 Feature catalogue (round 1)

All string similarities use `rapidfuzz` vectorised pairwise functions (`cpdist`, multi-threaded) or numba kernels, computed per shard.

**Name** (each on `norm`, `latin` and `skel`, and the best over DBA alternatives):
- Jaro-Winkler, normalised Levenshtein, Indel ratio, token_sort_ratio, token_set_ratio, partial_ratio.
- Monge-Elkan (JW inner).
- IDF-weighted token Jaccard and soft-TF-IDF (IDF per country partition, so rare tokens count more).
- Char 3-gram TF-IDF cosine.
- Exact core match; exact sorted-core match; first-token match; length ratio.
- **Asymmetric coverage:** fraction of S1 core tokens found in the record (fuzzy); fraction of record tokens explained by S1 tokens, known affixes or legal forms.
- Legal form: {same, both absent, one absent, different class}.
- Domain: record is domain-style; domain = joined core; domain ⊇ core prefix; domain = initials (+ suffix token).
- Script: record name Indic-script; romanisation used; share of tokens resolved by the mined dictionary vs the rule-based romaniser.

**Address:**
- Per aligned component type (street, locality, city, state, district): best similarity, and whether it is present in S1, in the record, or both.
- **Numbers:**
  - House-number sequence exactly equal; first part equal; parts matched part by part.
  - Numeric absolute difference of the first part; edit distance between the digit strings.
  - Digit-token Jaccard (excluding PIN and phone).
  - bis/ter agreement; unit agreement.
- PIN/postcode: equal / one missing / different. Phone: equal / one missing / different.
- Landmark similarity (if both have one).
- Whole address: char TF-IDF cosine, bag-of-tokens Jaccard, token_set_ratio.
- Component counts: matched, extra, missing. Flags for `null`/`N/A`.

**Ambiguity and competition** (computed after the pre-ranker, from candidate lists):
- Rank of the record in S1's list and gap to the best; rank of the S1 in the record's list and gap to the record's best S1.
- How many S1s in the partition share this S1's core name; how many share it in the same city.
- **How many S1s share this S1's exact normalised address** (5.46% of S1s share one).
- Block sizes of the keys that produced the pair; channel flags (which channels found it, and hop2).

**Other:**
- Source (S2 vs S3). Embedding cosine (if D1 exists). String lengths and token counts.
- **No country feature.** Country is only used to choose the parsing profile and the threshold set.

### 8.2 "What changed" (diff-signature) features

Planted traps and noisy true matches can score similarly (≈0.95) on every generic similarity. What separates them is *which kinds of edits* turn S1 into the record. The data generator uses one mix of edits for noise and another for traps (the Catalpa Bluff trap is *legal-form change + house-number change + abbreviation*). We make those edits explicit:

1. **Align tokens** within each field (name core, legal form, each address component). Build an S1-token × record-token similarity matrix and take a greedy maximum-weight alignment (numba).
2. **Label each aligned pair** as one of: `exact`, `case/accent-only`, `known-abbrev` (mined table), `script` (skeleton-equal across scripts), `typo` (edit distance ≤ 2 and length ≥ 4), `number-format` (`0684↔684`, `45th↔45nd`), `number-changed`, or `different-word`.
3. **Label each unaligned S1 token** as missing: `missing-legal`, `missing-number`, `missing-generic`, `missing-rare`.
4. **Label each unaligned record token** as extra: `extra-known-affix` (mined), `extra-legal`, `extra-number`, `extra-rare`.
5. **Features:** counts of each type per field, plus a few combinations: `number-changed ∧ legal-changed`, `number-changed ∧ street-exact`, `name-exact ∧ city-different`.

The same labels give automatic buckets for error analysis (§18) and for stratified hard-negative sampling (§8.4).

### 8.3 Round-1 model

- **LightGBM**, binary logloss. The CPU version on a 64-core machine is fast enough; the GPU version is optional. About 300 features, `num_leaves` 255, learning rate 0.05, early stopping on out-of-fold log-loss.
- **Training data:** candidate pairs of about 800k training S1s per fold, **subsampled by entity**. That is roughly 30–40M rows of float32 features. Positives come from the ground truth; negatives are all other candidates, which are naturally hard.
- **Out-of-fold predictions** for all training pairs come from 5-fold GroupKFold. The test prediction is the average of the 5 fold models.
- **Ablation:** monotone constraints (+1) on about 10 core similarity features, e.g. address TF-IDF cosine, house-number exact, core-name JW. This may cost a little in-distribution accuracy but can make France extrapolation safer. We decide by comparing leave-one-country-out results (§10.5).

### 8.4 Cross-encoder

- **Model:** `BAAI/bge-reranker-v2-m3` (568M, Apache-2.0) as the main choice, or `microsoft/mdeberta-v3-base` (≈280M, MIT) if throughput is tight. Both are multilingual, so they cover French and Indic scripts.
- **Input:** `S1: {name} | {address} [SEP] S{2|3}: {name} | {address}`, raw text on both sides. An ablation appends the `latin` view of the record.
- **Training data:** all positives plus hard negatives sampled from Stage-1 candidates at 1:3. Negatives are stratified by diff signature so every trap family is represented:
  - same core name, different address
  - same address, different name
  - house number differs by one part
  - only the legal form differs
  - generic name in the same city
  - skeleton-equal but different entity
- **Cross-fitting:** two folds. Each half's model scores the other half's out-of-fold pairs; the test set is scored by the average of both.
- **Which pairs to score:** round-1 `p ∈ (0.02, 0.98)` plus the top-3 per S1, estimated at 30–40% of candidates. Pairs outside the band get a "not scored" flag and the round-1 probability passed through.
- **Training settings:** bf16, max length 128, 1 epoch. Rough speed on one 48 GB-class GPU is a few thousand pairs/s for inference, so tens of millions of pairs take hours, not days. **Measure the real rate on day 1 and size the band from it.**

### 8.5 Round-2 model: group consensus and competition

Trained on **out-of-fold** round-1 (+ cross-encoder) scores, so confidence is realistic and errors don't feed themselves.

For a pair (e, r), let `M(e) = {r' ≠ r : p1(e,r') ≥ τ}` be the confident members (τ ≈ 0.8). Features:
- |M(e)|; max and mean name and address similarity between r and M(e).
- Agreement of r's house-number sequence and PIN with the **majority value** in M(e) ∪ {e}.
- Whether r is a kNN neighbour of any member of M(e).
- **Competitors:**
  - r's best other S1: its p1, its |M|, and the gap to e.
  - The softmax of p1 over r's candidate S1s, with a learned "none" logit taken from the ambiguity features.
- All round-1 features and scores are passed through.

**Record↔record similarities** are only computed between the top-8 candidates of each S1: at most 28 pairs per S1, about 50M in total for test, which is cheap with `rapidfuzz.cdist`.

**Output.** Round-2 probability, calibrated with isotonic regression fitted on out-of-fold predictions **per country profile**. France uses the US calibrator until the leaderboard suggests otherwise.

---

## 9. Stage 3: Decision layer

### 9.1 Ownership

For each record r, keep only the S1 with the highest round-2 probability. Set its probability to zero for every other S1: `q(e, r) = p2(e, r)` if `e = argmax_e' p2(e', r)`, else `0`.

This is exact given the verified structure (0 of 7.64M records have two owners). There is **no hard margin rule**. The ambiguity is already in p2 through the competitor features, so a true owner's only match is not dropped just because a look-alike scored close.

### 9.2 Entity gate

A LightGBM classifier per S1 with target `y = 1` if e has ≥ 1 true match. Trained on out-of-fold features. Features:
- `q` of the top-1, top-2 and top-3 candidates; the gap between top-1 and top-2.
- Sum of q; counts with q ≥ {0.3, 0.5, 0.8}; how many records have e as their best S1; number of candidates.
- Best name and address similarity among candidates; share of candidates from each channel.
- How often e's core name appears; how many S1s share its address; the competitors' strength for e's top-1 record.
- Output `g(e) = P(e has ≥ 1 match)`, calibrated with isotonic regression.

### 9.3 Set selection

```python
def select(e, cands, gate, T):             # T = thresholds for e's country profile
    C = sorted([(r, q[e, r]) for r in cands[e] if q[e, r] > 0], key=lambda x: -x[1])
    if not C:
        return []
    p_single = 1.0 - gate[e]
    r1, q1 = C[0]
    if q1 * T.kappa <= p_single or q1 < T.t_min:     # first-member rule (§3.1)
        return []
    S, n_src = [r1], {src(r1): 1}
    for r, q_r in C[1:]:
        k = len(S)
        if q_r < T_ADD_BASE[min(k, 6)] + T.delta:    # §3.2 table + tuned offset
            break
        if n_src.get(src(r), 0) >= CAP[src(r)]:      # S2 ≤ 5, S3 ≤ 6
            continue
        S.append(r); n_src[src(r)] = n_src.get(src(r), 0) + 1
    return S
```

`T_ADD_BASE = [–, 0.727, 0.759, 0.771, 0.778, 0.782, 0.785]`. `kappa` stands for E[F | top-1 correct] (≈ 0.7–0.9) and `t_min` is a safety floor (≈ 0.05–0.2).

### 9.4 Tuning

- **Grid search over (`kappa`, `t_min`, `delta`) per country profile** on the out-of-fold predictions for all 2.2M training S1s. The objective is the exact macro-F0.5.
- **Robustness check:** tune under both the normal setting and the unmatched-density stress test (§10.4). Pick the thresholds that maximise the **worse** of the two, unless §10.4 rejects the hypothesis.
- **France** starts with the US thresholds: Latin script, addresses that start with the number, and similar legal-form conventions. It moves only on leaderboard evidence (§12).
- **Alternative to compare against:** exact expected-F0.5 maximisation over prefixes, using the calibrated q and the gate for the empty-set probability. Keep it only if it beats the rule above out-of-fold.

---

## 10. Validation protocol

### 10.1 Folds

- `fold(e) = hash(S1 id) mod 5`. Records inherit their owner's fold; unmatched records have no fold and only appear as negatives.
- A pair (e, r) belongs to fold(e). All out-of-fold models (pre-ranker, round 1, round 2, gate) use these folds.
- The two-fold cross-fitting for neural models uses halves aligned with the folds: {0, 1, 2} vs {3, 4}.
- **Blocking needs no folds,** because TF-IDF and keys are unsupervised. The only supervised part is the mined tables, which are leak-safe by their support threshold (§6.6).
- The decision layer is tuned on **all** training S1s at once, using out-of-fold scores. The one-owner competition therefore happens at full density with realistic confidence.

### 10.2 Metric implementation

We use the exact metric (Appendix A), tested against the worked example in the problem statement (0.714) and the singleton rules. Submission formatting is checked with `utils/validate_submission.py` on every run.

### 10.3 Reports (every run)

- Macro-F0.5 overall; per country; singleton vs non-singleton; by true group size; by source.
- Micro precision and recall; best-achievable score; candidates per S1.
- Top-50 false positives and false negatives with their diff-signature buckets.

### 10.4 Unmatched-density stress test

**Check first:**
- Compare the distribution of each record's best pre-ranker score in test vs train, per country. If test has a fatter low tail, that means more unmatched records.
- Compare it with the distribution after deleting 19% of training S1s.

**If confirmed, simulate.** Delete a random 19% of training S1s, so their records become unmatched. Recompute the competition features from the existing out-of-fold pair scores (grouped operations, cheap), then rerun round 2, the gate and the decision. For a check of the approximation, rerun Stages 1–3 in full on one partition with the reduced S1 set.

### 10.5 Leave-one-country-out

Train on US and evaluate on India, and vice versa. Any feature whose importance or effect collapses across countries is country-specific and a France risk. This measures **robustness in general, not France specifically**: France is closer to the US in script and address order.

### 10.6 S1↔S1 diagnostic (no labels, no training)

S1 is deduplicated, so every S1↔S1 pair is a true non-match.

1. Retrieve near-duplicate S1↔S1 pairs with the same blocking channels, per country: train US, train India and test France.
2. Score them with the pair model, treating one side as the record.
3. Compare the distribution of false-positive scores across countries.

If French non-matches score higher than US/India ones, raise the France thresholds. S1↔S1 pairs are clean on both sides, so read this as a **relative** signal only.

---

## 11. France strategy (15% of test, no labels)

1. **Parsing profile** (§6.5): Rue/R./R, Av/AV, Bd, Imp, Allée, Ch, `St` = Saint, bis/ter/`B`, `(41)`, `NO.` prefix, département ↔ region as a soft mismatch, SARL/SAS/SASU/EURL/SCI/SA/SNC with dotted forms, *Cie*, French stopwords, `& = et`.
2. **Country-agnostic features only,** with no country categorical; optional monotone constraints (§8.3).
3. **Multilingual neural components** (cross-encoder; dense model if used), which already understand French.
4. **Thresholds** start from the US profile; the S1↔S1 diagnostic and paired leaderboard probes (§12) adjust them.
5. **Test-time pseudo-labelling of France is not used** unless the organisers confirm in writing, through the query form, that unsupervised use of test inputs is allowed. We ask on day 1.

---

## 12. Leaderboard plan (15 submissions)

Two submissions that differ only in some rows give a **paired difference**. Only the changed entities contribute to that difference, so it has low variance on a large public subset. We use paired probes for questions validation cannot answer, and never for fine-tuning in-distribution thresholds.

| Day | # | Submission | Purpose |
|---|---|---|---|
| 25 | 1 | Baseline v1: A1+N1+keys blocking, round-1 LightGBM, one owner per record, simple gate | Valid end-to-end run; measures the gap between out-of-fold and leaderboard |
| 25 | 2 | **All empty** | Public singleton rate (a prior for the gate) |
| 25 | 3 | (spare) v1 with thresholds from the stress test | Tests the unmatched-density hypothesis on the leaderboard |
| 26 | 4 | v2: diff signatures, parsing fields, gate, tuned thresholds | Main progress |
| 26 | 5 | v2 with **France rows empty** | `F_France ≈ s_F + (LB₄ − LB₅)/w_F`, where w_F ≈ France share of the public subset (≈0.15) and s_F ≈ its singleton rate (≈0.056) |
| 26 | 6 | v3: + cross-encoder + round 2 | Main progress |
| 26 | 7 | v3 with France δ + 0.05 (stricter) | Direction for France thresholds |
| 26 | 8 | (spare) v3 with France δ − 0.05 or monotone-constrained | Brackets the France optimum |
| 27 | 9–12 | v4 ensemble / final threshold variants | Final selection |
| 27 | 13 | **Final**: best out-of-fold configuration + leaderboard-informed France thresholds | Make the **last upload** the one we would bet on, in case the last submission counts |

The formula for probe 5 assumes the public subset is a random sample of test S1s.

---

## 13. Models, licences and parameter budget

| Role | Model | Parameters | Licence | Status |
|---|---|---|---|---|
| Pre-ranker, round 1, round 2, gate | LightGBM | – (trees) | MIT | Core |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | 568M | Apache-2.0 | Core (day 2) |
| Cross-encoder (lighter alternative) | `microsoft/mdeberta-v3-base` | ≈280M | MIT | Fallback |
| Bi-encoder (conditional) | `intfloat/multilingual-e5-base` / `Qwen/Qwen3-Embedding-0.6B` | 278M / 0.6B | MIT / Apache-2.0 | Only if §7.6 shows a gap |
| **Total** | | **≤ 1.2B** | | ≪ 8B under the strict reading |

Llama, Gemma and similar custom-licence models are excluded. No 7B+ model is used.

Supporting libraries (all permissively licensed): polars, pyarrow, rapidfuzz, numba, scikit-learn, scipy, sparse_dot_topn, lightgbm, torch, transformers, sentence-transformers, faiss (if the dense channel is used), indic-transliteration / anyascii, jellyfish.

---

## 14. AWS compute plan

| Box | Instance (or equivalent) | Used for |
|---|---|---|
| CPU box | `r7i.16xlarge` (64 vCPU, 512 GiB), or `m7i.16xlarge` (256 GiB) | Ingest, normalisation, TF-IDF blocking, features, LightGBM, decision tuning |
| GPU box | `g6e.2xlarge` (1× L40S 48 GB) ×1–2; `p4d`/`p5` only if quota allows | Cross-encoder training and inference; dense model if used |
| Storage | S3 bucket (artifacts, checkpoints, Parquet) + 1 TB gp3 EBS per box | Stage outputs, cached features |

**First action on day 1:** request service-quota increases for *Running On-Demand Standard instances* (vCPU) and *Running On-Demand G and VT instances*. New accounts often start near zero for GPU families, and approval can take hours. **The pipeline must reach a full submission with no GPU at all**; the cross-encoder is strictly additive.

**Environment:**
- Create the same conda environment named `ml` on every instance from a pinned `environment.yml` / `requirements.txt`.
- Run everything in `tmux`. Every stage writes Parquet shards and a `_SUCCESS` marker to S3, so a lost instance only costs the current shard.
- Use on-demand instances (not spot) for critical-path stages on day 3.

**Memory sizing:**
- About 60–80M candidate pairs (train + test) × about 300 float32 features ≈ 70–100 GB in total, kept as shards.
- LightGBM trains on an entity subsample of about 30–40M rows (≈ 40–50 GB), which fits in 512 GiB.

---

## 15. Timeline (IST)

Workstreams: **A** = data, blocking and features (critical path, CPU box). **B** = neural (GPU box). **C** = validation, decision and submissions. If working solo, A and C come first, and B runs in the background.

### Day 1 (25 Sep): a valid, measured baseline

- **A:** quota requests; instances; data to S3 → Parquet, split by country; profile script (recompute §2); normalisation v1 (`norm`, `latin`, `skel`, legal form, number sequences, PIN/phone); channels A1, N1, A2/N2 v1; best-achievable-score evaluator.
- **C:** exact metric + folds; pre-ranker; round-1 LightGBM on similarity + number features; ownership + simple gate + §3 thresholds. **Submissions 1 and 2 (baseline, all-empty).** Submit the day-1 query to the organisers (8B reading; unsupervised use of test inputs).
- **B:** cross-encoder data pipeline; measure its throughput; start cross-encoder fold-1 training once round-1 out-of-fold candidates exist.
- **Go/no-go:** best-achievable score ≥ 0.98 by the end of day 1. If not, day-1 evening goes into blocking, following the §18 buckets.

### Day 2 (26 Sep): the big accuracy gains

- **A:** diff-signature features; component alignment; landmark, phone and municipal-number fields; DBA and domain features; France profile; 2-hop expansion (if it raises the best-achievable score).
- **C:** entity-gate model; round-2 consensus model; stress-test hypothesis check (§10.4); threshold tuning; S1↔S1 diagnostic. **Submissions 4–8.**
- **B:** cross-encoder on both folds → out-of-fold scores + test scores on the band; dense model only if the §7.6 analysis shows a gap.

### Day 3 (27 Sep): harden, freeze, ship

- **Morning:** seed ensemble (3 LightGBM seeds for rounds 1 and 2); final calibration; France thresholds from probes 5–8; error-analysis fixes only if they are small and validated.
- **16:00 IST: code freeze.** A clean end-to-end run from raw TSVs on a fresh instance with the pinned environment produces both TSVs; validate.
- **Evening:** package the zip (`output/`, `code/business_entity_resolution/`, `Documentation_template.md` filled in). **Final submissions by 22:00 IST**, leaving a buffer before 23:59.

---

## 16. Code layout and reproducibility

The implementation lives in `code/business_entity_resolution/`, the folder the challenge requires in the submission zip. Its `README.md` has the exact commands and AWS instance guidance.

```
code/business_entity_resolution/
├── README.md, requirements.txt, environment.yml     # pinned environment (conda env "ml")
├── configs/default.yaml, configs/smoke.yaml         # every knob documented inline
├── scripts/make_sample.py, score_sample.py, package_submission.sh
└── src/ber/
    ├── io.py                        # TSV -> Parquet, unified uid space, submission writers
    ├── normalize/                   # text.py, translit.py (Indic romaniser + skeleton), profiles.py,
    │                                #   names.py, address.py, tables.py, run.py (parallel stage 0)
    ├── mining/tables.py             # §6.6 three-pass table mining
    ├── blocking/                    # vectors.py (hashed TF-IDF + RP), knn.py (exact GPU/CPU top-k),
    │                                #   keys.py, candidates.py (union), prerank.py (OOF + floor), expand.py (2-hop)
    ├── features/                    # pair.py (round-1, sharded), diffsig.py, context.py, consensus.py (round-2)
    ├── models/                      # gbdt.py (LightGBM/XGBoost + OOF), calibrate.py, rounds.py (r1/r2),
    │                                #   cross_encoder.py, biencoder.py, gate.py
    ├── decision/                    # select.py (vectorised set selection), tune.py (grid + predict)
    ├── eval/                        # metric.py (exact macro F0.5 + best achievable score), stress.py, s1s1.py
    ├── eda/profile.py               # recomputes every §2 statistic
    └── pipeline/                    # run.py (CLI, resumable stages), outputs.py (TSVs + validator)
```

**CLI stages, in order:**

`ingest → eda → mine → normalize → dense → block → prerank → expand → features → r1 → ce_train → ce_infer → r2 → gate → tune → predict → outputs`

Extra stages: `s1s1` and `stress_check`.

**Two reproduction paths:**
1. **Full:** `python -m ber.pipeline.run --config configs/default.yaml --stage all` trains everything from the raw TSVs.
2. **Inference-only:** restore `work/tables.pkl` and `work/models/` (the models bundle from `scripts/package_submission.sh <team> --with-models`), then run with `--set run.inference_only=true --stage all`. This processes the test split only and loads every model, calibrator, floor and threshold. It is verified to reproduce the full run's output files byte for byte.

**Rules (as implemented):**
- Seeded RNGs; LightGBM runs with `deterministic` + `force_col_wise`.
- Every model input and ranking is sorted by `(s1_uid, rec_uid)` with explicit tie-breaks; cuBLAS is pinned via `CUBLAS_WORKSPACE_CONFIG`.
- Two from-scratch runs give byte-identical `matching_results.tsv` and `candidate_pairs.tsv`.
- Every stage writes a completion marker, so the pipeline resumes after interruption. `--force` re-runs a stage and `--from <stage>` continues from it.
- `validate_submission.py` runs automatically in `outputs`. The code also asserts that every match is one of the candidates.
- Leaderboard probes (`--probe <name>` with `decision.overrides`) write to `output/probes/<name>/` without touching the main submission.

### 16.1 Verification status (25 Sep 2026)

| Component | Status |
|---|---|
| Stages 0–3 on CPU (normalise, mining, blocking, pre-ranker, 2-hop, features, r1, r2, gate, stress, tuning, outputs) | **Run end to end** on a 5k-S1 sample built from the training data (`scripts/make_sample.py`). The official validator passes. Pseudo-test macro F0.5 is 0.9976 against held-out truth (the sample has random distractors, so this is not a leaderboard estimate). |
| Reproducibility and the inference-only path | **Verified** byte-identical |
| Leaderboard probes (all-empty, France-empty, France δ-shift), stress simulation, S1↔S1 diagnostic, forced 2-hop | **Run** |
| XGBoost fallback | **Run** (fit, save, reload, identical predictions) |
| Cross-encoder and bi-encoder | **Plumbing verified with a stub model** (training loop, cross-fitting, band, sharded inference, round-2 integration). Not yet run with the real Hugging Face weights, because the local environment has `transformers` 2.1.1. Run it on the AWS box with the pinned `transformers==4.46.3`. |
| Full-scale runtime and memory | **Not yet measured.** Sizing in §14 and the README is estimated. |

## 17. Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| GPU quota not granted | Quota request still pending by afternoon of day 1 | CPU-only pipeline (LightGBM + diff signatures) is complete; drop the cross-encoder |
| Blocking best-achievable score < 0.98 | §7.6 report | Look at the miss buckets; add dense D1 or widen A1/N1 k |
| Stress-test hypothesis wrong | §10.4 distributions don't match | Tune on the normal setting only |
| Out-of-fold vs leaderboard gap > 2 points | Submission 1 | Check format and IDs; check whether France accounts for the gap (probe 5); check the singleton rate from probe 2 |
| France collapse | Probe 5 gives F_France ≪ US | Stricter France δ; monotone model; heavier weight on address features for France |
| Out of memory on feature building | Instance metrics | Smaller shards; float32/uint8 features; stream to Parquet |
| Validator warning (match not in candidates) | `validate_submission.py` | The decision layer only removes; a unit test asserts matches ⊆ candidates |
| Rule dispute (8B, test use) | Organiser reply | Already on the strict reading; no pseudo-labels |
| Missing the deadline | Timeline slips | Freeze at 16:00 on day 3; always have the latest validated submission ready |

---

## 18. Error-analysis playbook (after every out-of-fold run)

1. **Sort false positives by p, bucketed by diff signature:**
   - same name, different address
   - sibling number (number changed, street exact)
   - same address, different name
   - generic name in the same city
   - legal-form-only difference
   - cross-script look-alike
2. **Sort false negatives by type:**
   - not in the candidate set (blocking)
   - Indic name, domain name, trade name
   - empty address
   - house number changed in a true match
   - district or département injected
   - lost the ownership competition to a look-alike
3. **Target the largest bucket** with one change: a new feature, a new parsing rule, or a threshold. Re-measure. Keep the change only if out-of-fold macro-F0.5 improves overall **and** does not drop on singletons.
4. **Log every change** with its before/after score in the run manifest. This feeds §5 of the methodology document.

---

## 19. Mapping to the Documentation template

| Template section | Source in this document |
|---|---|
| 2.1 Problem analysis | §2 (verified facts), §3 (metric) |
| 2.2 Solution strategy | §0, §5 |
| 3. Candidate generation | §7 (channels, floor, 2-hop, best-achievable score, candidate counts) |
| 4. Matching model | §6 (views/parsing), §8 (features, diff signatures, round 1 / cross-encoder / round 2), §9 (thresholds) |
| 5. Results and error analysis | §10.3 reports, §18 buckets |
| Appendix A | §16 |

---

## Appendix A: Metric and best-achievable-score code

```python
def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def macro_f05(pred: dict, truth: dict) -> float:          # keys: every S1 id
    return sum(f05(pred.get(e, set()), truth[e]) for e in truth) / len(truth)

def ceiling(cands: dict, truth: dict) -> float:           # best achievable from candidates
    return sum(f05(truth[e] & cands.get(e, set()), truth[e]) for e in truth) / len(truth)

# Unit test from the problem statement:
assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.714) < 1e-3
```

## Appendix B: Threshold t(k) for adding a member

Suppose k members are selected and assumed correct, and a candidate has probability p of also being a true match.

- If it is **added**: true → F = 1; false → P = k/(k+1), R = 1, so F = a = 1.25k / (1.25k + 1).
- If it is **not added**: true → P = 1, R = k/(k+1), so F = b = 1.25k / (1.25k + 0.25); false → F = 1.

Add iff `p + (1−p)·a > p·b + (1−p)`, i.e. `p > (1 − a) / (2 − a − b)`. This gives 0.727 (k=1), 0.759, 0.771, 0.778, 0.782, 0.785 (k=6), approaching 0.8.

For the first member (§3.1), the comparison is with P(singleton), not with P(the candidate is wrong), because a wrong top-1 and an empty prediction both score 0 on entities that have matches.

## Appendix C: How the §2 statistics were computed

- **Row counts and country counts:** full-file passes over each TSV.
- **Group-size and per-source distributions, singleton rates:** a full pass over `train_ground_truth.tsv` joined to `train_source1.tsv` by ID.
- **One-owner check:** every ID in `matched_entity_ids` across all ground-truth rows counted; 7,638,365 IDs, none repeated.
- **S1 shared names and addresses:** lowercase, remove non-alphanumerics (and spaces for addresses), count duplicates.
- **Noise-prevalence table:** byte-level regex over the name or address column. Indic = UTF-8 lead bytes for U+0900–U+0DFF; domain = `.com | www. | .in | .fr | .net | .org`; empty = empty address field; null = `null | n/a | none` tokens.
- ***Sample* statistics:** every 50th ground-truth row (44,136 S1s, 153,241 pairs); S1 and matched records joined by ID; ASCII-lowercase alphanumeric tokens for overlap.
- **Density hypothesis:** `(S2+S3)/S1` per country; the 19% deletion rate solves `10.32M / (2.207M · (1 − d)) = 5.754`.

All of these are reimplemented in `src/ber/eda/profile.py` (Polars) so the numbers can be regenerated.
