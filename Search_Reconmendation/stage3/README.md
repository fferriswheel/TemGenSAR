# Stage 3: Data and Prototype Preparation

> Historical prototype documentation. This page records early Amazon data preparation and smoke tests; its plans, paths, and placeholder tokenizers describe that stage only. For the current system, see [method](../../docs/method.md), [experiments](../../docs/experiments.md), [code navigation](../../docs/research-status.md), and [environment setup](../../environment/README.md).

> 本页是早期数据准备档案，下面的本地路径、待办状态和简化编码均指当时的原型。当前编码结构、更新方案和代号解释以[方法说明](../../docs/method.md)与[术语对照](../../docs/glossary.md)为准。

## 1. Goal

Stage 3 moves the project from method design to a runnable prototype.

The immediate goal is not to train the final dynamic model. The goal is to build a reproducible data and code base that can support:

```text
1. GenSAR-style public Search + Recommendation prototype on Electronics.
2. Our static [shared prefix | task suffix] SID prototype.
3. Later temporal-drift experiments on Amazon Reviews 2023 Electronics.
```

## 2. Current Local Assets

External code:

```text
/data/cjk/Search_Reconmendation/external/GenSAR
```

Public Amazon product search data:

```text
/data/cjk/Search_Reconmendation/external/Amazon-Product-Search-Datasets/reviews_Electronics_5.json.gz
```

Downloaded Electronics files:

```text
query_text.txt.gz
train.qrels.gz
test.qrels.gz
train_review_id.txt.gz
```

## 3. Electronics Data Snapshot

The public Electronics S&R data has:

```text
queries:           989
train qrels:       1,344,310
test qrels:        5,541
train review ids:  1,275,432
all qrels:         1,349,851
users:             192,403
items:             62,993
```

The qrels format is:

```text
<user_id>_<query_id> 0 <asin> <label>
```

Example:

```text
A26ZDKC53OP6JD_22 0 B000M51NRM 1
```

The query file contains category-derived pseudo queries, for example:

```text
electronics computer accessory interconnect ethernet cable
electronics computer accessory data storage external hard drive
electronics supplies audio accessory interconnect video hdmi cable
```

Important limitation:

```text
This public product-search data does not provide interaction timestamps.
It is suitable for Stage 3 S&R pipeline prototyping, not for the final temporal drift main experiment.
```

## 4. What GenSAR Expects

GenSAR assumes a processed dataset under:

```text
external/GenSAR/data/Amazon_Electronics
```

Important expected files:

```text
vocab/user_vocab.pkl
vocab/item_vocab.pkl
vocab/query_vocab.pkl
vocab/src_session_vocab.pkl
emb/bge-base-en-v1.5.pt
emb/UniSAR.pt
emb/<semantic_index_file>.json
emb/<collaborate_index_file>.json
```

Index training entry:

```text
external/GenSAR/index/run_index.py
```

Generative model entry:

```text
external/GenSAR/finetune/run_amazon.py
```

GenSAR's public repository contains code, but not ready-to-run processed Amazon data.

## 5. Stage 3 Execution Plan

### Step 1: Define Our Intermediate Data Format

Create a stable project-level format independent of GenSAR internals:

```text
items.jsonl
users.jsonl
queries.jsonl
search_interactions.jsonl
rec_interactions.jsonl
time_windows.json
sid_mapping.json
```

For the public Electronics prototype:

```text
queries.jsonl               from query_text.txt.gz
search_interactions.jsonl   from train/test qrels
rec_interactions.jsonl      from review order if raw review data is available
```

For Amazon Reviews 2023:

```text
items.jsonl                 from metadata
rec_interactions.jsonl      from reviews with timestamp
queries.jsonl               from category/title/attribute pseudo-query extraction
search_interactions.jsonl   from pseudo-query to item links
time_windows.json           from timestamps
```

### Step 2: Build Electronics Search Prototype Data

Convert the public Electronics qrels into:

```text
search_interactions.jsonl
queries.jsonl
item_ids.jsonl
```

Each search interaction should contain:

```json
{
  "user_id": "...",
  "query_id": 22,
  "query": "...",
  "item_id": "B000M51NRM",
  "label": 1,
  "split": "train"
}
```

### Step 3: Recover / Build Recommendation Sequences

There are two options:

```text
Option A:
  Use the original Amazon review file referenced by train_review_id.txt.gz
  to recover chronological recommendation sequences.

Option B:
  For the first prototype only, derive weak user-item sequences from qrels.
  This is not ideal for final experiments, but enough to test data plumbing.
```

Recommended route:

```text
Use Option B for a smoke-test prototype.
Use Option A before any serious experimental result.
```

### Step 4: Implement Static Modular SID Prototype

First prototype:

```text
Search SID = [P1, P2 | S1, S2]
Rec SID    = [P1, P2 | R1, R2]
codebook size = 256
```

Minimum static tokenizer:

```text
1. Build search-side text/query representation.
2. Build rec-side collaborative representation.
3. Learn shared prefix via gated fusion.
4. Learn task suffix via residual quantization.
5. Export SID mapping JSON.
```

### Step 5: Connect to Generative Training

Use a small subset first:

```text
10K users
10K items
all available queries for selected items
```

Target tasks:

```text
Search:
  query/context -> [P | S]

Recommendation:
  user history -> [P | R]
```

### Step 6: Move to Amazon Reviews 2023 Electronics

Once the static S&R prototype works, move to temporal data:

```text
1. Download Amazon Reviews 2023 Electronics reviews and metadata.
2. Build chronological rec sequences.
3. Generate pseudo queries from metadata.
4. Define time windows T0...Tk.
5. Run static model on T0.
6. Add drift diagnosis and selective refresh.
```

## 6. Immediate Next Tasks

1. Create `scripts/convert_product_search_electronics.py`. Done.
2. Convert `query_text.txt.gz`, `train.qrels.gz`, and `test.qrels.gz` into JSONL. Done.
3. Produce dataset stats and sanity checks. Done.
4. Create a tiny sampled dataset for fast iteration. Done.
5. Decide whether to download original Amazon Electronics review data for chronological rec sequences. Decision below.

## 7. Sanity Check Result

Sanity checker:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/check_product_search_electronics.py
```

Report:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sanity_report.json
```

Main findings:

```text
bad rows: 0
train/test exact user-query-item overlap: 0
test users in train ratio: 1.0
test items in train ratio: 1.0
test queries in train ratio: 0.0
```

Interpretation:

```text
The public Electronics product-search data is clean for search-side prototyping.
The test split evaluates unseen query generalization because test queries do not appear in train.
Users and items in test are all covered by train.
```

Qrels-derived recommendation sequences:

```text
users with sequence length >= 2:  192,403
users with sequence length >= 5:  141,161
users with sequence length >= 10: 28,878
median sequence length: 5
mean sequence length:   6.99
```

Important caveat:

```text
These sequences use qrels file order, not chronological order.
They are acceptable for a smoke-test dataloader, but not for serious recommendation or drift experiments.
```

## 8. Review Data Decision

For a serious joint S&R prototype, we need the original Electronics 5-core review file because it contains chronological review interactions and can be aligned with `train_review_id.txt.gz`.

Expected source:

```text
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
```

Alternative URL:

```text
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
```

Expected size:

```text
495,854,086 bytes
```

If downloaded manually, place it at:

```text
/data/cjk/Search_Reconmendation/stage3/raw/amazon_legacy_5core/reviews_Electronics_5.json.gz
```

Metadata source:

```text
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
```

Expected metadata size:

```text
186,594,679 bytes
```

If downloaded manually, place it at:

```text
/data/cjk/Search_Reconmendation/stage3/raw/amazon_legacy_5core/meta_Electronics.json.gz
```

Current status:

```text
Review file downloaded manually by user and verified locally.
Review file size matches expected size.
Review gzip integrity check passed.
Raw review count: 1,689,188.

Metadata file downloaded manually by user and verified locally.
Metadata file size matches expected size.
Metadata gzip integrity check passed.
```

Important alignment note:

```text
train_review_id.txt.gz is not line-by-line aligned with train.qrels.gz.
It is a set/list of original review line ids used for training reviews.
Because qrels are expanded by query, one review may correspond to multiple qrels rows.
```

Correct usage:

```text
Use train_review_id.txt.gz to filter original review lines.
Then sort each user's reviews by unixReviewTime to build recommendation sequences.
```

## 9. Smoke-test Dataloader

Smoke-test example builder:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/build_smoke_dataloader_examples.py
```

Generated files:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/search_smoke_examples.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/rec_smoke_examples.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/mixed_smoke_examples.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/smoke_stats.json
```

Current smoke data:

```text
search examples: 10,000
rec examples:    10,000
mixed examples:  20,000
max history len: 5
```

Smoke dataloader:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/smoke_dataloader.py
```

Example command:

```bash
python /data/cjk/Search_Reconmendation/stage3/scripts/smoke_dataloader.py --batch-size 3 --num-batches 2
```

The dataloader returns:

```text
task
input_text
target_text
target_item_id
```

Example task formats:

```text
Search:
  input_text  = <SEARCH> electronics supplies audio accessory interconnect video hdmi cable
  target_text = <P1_126> <P2_198> <S1_234> <S2_228>

Rec:
  input_text  = <REC> B000M51NRM B007KFVQXI
  target_text = <P1_69> <P2_138> <R1_112> <R2_173>
```

Important caveat:

```text
The current target SID is a deterministic placeholder hash, not a learned modular SID.
It exists only to test data flow before implementing the real tokenizer.
```

## 10. Chronological Recommendation Data

Chronological rec sequence builder:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/build_legacy_review_rec_sequences.py
```

Generated files:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/rec_interactions_chrono.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/rec_chrono_stats.json
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/rec_chrono_smoke_examples.jsonl
```

Chrono rec stats:

```text
train reviews loaded: 1,275,432
users:                192,403
items:                62,993
median seq length:    5
mean seq length:      6.63
users len >= 5:       133,309
users len >= 10:      25,125
```

For recommendation smoke tests, prefer:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/rec_chrono_smoke_examples.jsonl
```

over the older qrels-order file:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/rec_smoke_examples.jsonl
```

## 11. Baseline Modular SID Smoke Tokenizer

Metadata-to-items converter:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/build_items_from_legacy_metadata.py
```

It will generate:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/items.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/items_stats.json
```

Current status:

```text
items.jsonl generated successfully.
metadata items needed: 62,993
metadata items found:  62,993
metadata missing:      0
```

Baseline SID tokenizer:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/build_baseline_modular_sid.py
```

Smoke target item subset:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/smoke_target_item_ids.jsonl
```

Generated smoke SID mapping:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_smoke_meta/sid_mapping.baseline.json
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_smoke_meta/sid_mapping.baseline.stats.json
```

Current smoke SID mapping stats:

```text
items:                  12,133
metadata used:          true
prefix unique count:    8,812
search SID collision:   4.89%
rec SID collision:      7.16%
```

Important caveat:

```text
This baseline tokenizer uses query aggregation, user aggregation, item_id tie-break tokens,
TF-IDF / HashingVectorizer, and MiniBatchKMeans.
It is only a smoke tokenizer and is not the final gated residual SID model.
```

SID applier:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/apply_sid_to_smoke_examples.py
```

Generated SID-based smoke examples:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/search_smoke_examples.sid.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/rec_chrono_smoke_examples.sid.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/mixed_smoke_examples.sid.jsonl
```

These files are now the preferred inputs for smoke-test generative dataloading.

## 12. Modular SID v1

v1 tokenizer:

```text
/data/cjk/Search_Reconmendation/stage3/scripts/build_modular_sid_v1.py
```

Core idea:

```text
1. Build shared text representation from title, category, brand, and query terms.
2. Cluster shared representation into shared prefix [P1, P2].
3. Build search representation from title, description, and query terms.
4. Build rec representation from chronological sequence-neighbor item context.
5. Residualize search / rec representations by subtracting prefix-bucket centroids.
6. Cluster residuals into task suffixes [S1, S2] and [R1, R2].
```

Recommended smoke configuration:

```text
prefix_k1 = 128
prefix_k2 = 32
search_k1 = 256
search_k2 = 256
rec_k1 = 256
rec_k2 = 256
neighbor_window = 2
svd_dim = 128
```

Generated files:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v1_smoke_p128_32/sid_mapping.v1.json
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v1_smoke_p128_32/sid_stats.v1.json
```

Smoke v1 stats:

```text
items:                    12,133
prefix unique count:      3,719
prefix bucket p50 / p95:  2 / 11
prefix max bucket size:   58
search SID collision:     8.15%
rec SID collision:        0.52%
NMI(prefix, search suffix): 0.801
NMI(prefix, rec suffix):    0.782
```

Generated v1 smoke examples:

```text
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/search_smoke_examples.sid_v1.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/rec_chrono_smoke_examples.sid_v1.jsonl
/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/mixed_smoke_examples.sid_v1.jsonl
```

Current caveat:

```text
v1 has real prefix-centroid residual suffixes, but the NMI between prefix and suffix is still high.
This means the residual suffix is not fully disentangled yet.
The next tokenizer iteration should improve decoupling, for example by clustering suffixes within prefix buckets,
adding stronger residual normalization, or moving to the gated neural v2 tokenizer.
```
