# MovieLens-25M Temporal Search+Rec Pilot v1

## 1. Benchmark

- Fixed catalog: 18,191 movies with a positive interaction before 2015-01-01.
- T0 Rec: 8,721,602 positive ratings; future Rec: 3,270,308 positives.
- Temporal split: ten six-month windows, W1-W10 (2015H1-2019H2).
- Search: metadata-derived pseudo-query task held stationary in this pilot, so it is a control for Rec-only drift and task interference rather than natural search drift.
- Evaluation: one ground-truth item plus 99 sampled negatives.

## 2. T0 SID

The v13b/M7b family uses `[shared_1, shared_2 | task_1, task_2]` with four 256-entry codebooks. Common, Search, and Rec views are produced from MovieLens metadata, pseudo-query supervision, and T0 behavior respectively. CG-Sinkhorn yields zero Search and Rec collision at T0.

Raw suffix collision before CG-Sinkhorn was 0.53% for Search and 3.71% for Rec by duplicate fraction. The final operational SID has zero collision for both tasks.

## 3. W0 to W1 update

- The Rec drift detector selected 150 of 3,991 eligible items.
- The Rec representation was warm-started using W1 behavior; the Search encoder was frozen.
- Only the Rec task adapter was trained: 14,528 trainable parameters and 7.82 seconds.
- Sticky reassignment changed 23 Rec SIDs (0.126% of the catalog).
- Shared prefixes and every Search SID were exactly preserved.
- Rec collision remained zero after the update.

The downstream incremental set contains 43,882 W1 Rec rows, 1,000 Search replay rows, and matched Item2Index/Index2Item supervision for the same 23 items. Both dynamic and fixed-SID control arms use one epoch, 351 optimization steps, and the same random seed.

## 4. W2 results

| Arm | Rec HR@10 | Rec nDCG@10 | Search HR@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen T0 SID + T0 model | 0.44715 | 0.28512 | 0.7090 | 0.46161 |
| Updated SID, no T5 alignment | 0.44715 | 0.28473 | 0.7090 | 0.46161 |
| Dynamic SID + matched W1 training | 0.45161 | 0.28589 | 0.7064 | 0.45996 |
| Fixed SID + matched W1 training | 0.45161 | **0.28618** | 0.7064 | **0.46012** |

On the 237 W2 rows whose targets are among the 23 changed items, dynamic SID gives HR@10 0.08861 and nDCG@10 0.03554; the fixed-SID control gives 0.08861 and 0.03879.

## 5. Interpretation

The benchmark is useful: it provides ten natural Rec windows and a stationary Search control, and it exposes substantial temporal degradation without changing the catalog. The selective SID machinery also satisfies its structural contract and has negligible adapter cost.

The current `K=150, sticky=0.1` update is not yet a positive result. The overall Rec gain over the frozen model comes from ordinary downstream incremental training, because the matched fixed-SID control is slightly better. W1 validation favored dynamic SID by about 0.0010 Rec nDCG@10, but W2 reversed the ordering; this is consistent with fitting a transient W1 signal rather than learning a durable migration.

### Drift diagnostics

The pooled T0-to-W1 item-frequency comparison looks strong (TV 0.4261, JS divergence 0.2165, Spearman 0.6554, top-100 overlap 0.50), but this is partly a window-definition artifact: T0 pools all history before 2015 whereas W1 is only six months. A matched six-month comparison, 2014H2-to-2015H1, is moderate instead (TV 0.1472, JS 0.0460, Spearman 0.8590, top-100 overlap 0.90). W1-to-W2 is milder again (TV 0.0931, JS 0.0202, Spearman 0.8956, top-100 overlap 0.91).

The detector still finds a concentrated popularity shift: its 150 items account for 0.28% of pooled-T0 events, 9.63% of W1 events, and 8.48% of W2 events. However, their T0-to-W1 UniSAR embedding cosine distance averages only 0.000365. Thus the count-based detector sees drift that the suffix input representation barely transmits.

Sticky-budget scanning changes 54, 36, 27, 23, and 20 of the 150 selected items at switch ratios 0, 0.025, 0.05, 0.1, and 0.2 respectively. Even without a switching penalty, only 54 items cross code boundaries. Increasing the update budget alone is therefore unlikely to solve the representation/detector mismatch.

The next experiment should first redefine the detector's left side as a matched recent window (2014H2, not pooled T0) and make the Rec representation objective sensitive to window-specific transition/popularity change. Sticky ratio/change count should be tuned only after that alignment. Multi-seed and W3-W6 rollout should wait until one configuration beats the fixed-SID control on W2.

## 6. Artifacts

- Benchmark: `stage3/data/movielens25m_fixed18k_temporal_v1`
- GenSAR view: `stage3/data/movielens25m_gensar_temporal_v1`
- T0 SID: `stage3/artifacts/movielens25m_v13b_t0_pilot_v3_dim768`
- W1 Rec adapter: `stage3/artifacts/movielens25m_v13b_w1_rec_adapter_k150_v1`
- W1 operational SID: `stage3/artifacts/movielens25m_v13b_w1_rec_sticky_k150_v1`
- Dynamic run: `stage3/runs/movielens25m_temporal/W1_k150_align23_W2_eval`
- Fixed-SID control: `stage3/runs/movielens25m_temporal/W1_fixed_sid_control_align23_W2_eval`
