### Retrieval — `0113604596` (178 answerable questions)

| chunk profile | stage | recall@1 | recall@5 | recall@8 | mrr@20 | ndcg@10 | recall@2000tok | ctx tok@8 |
|---|---|---|---|---|---|---|---|---|
| sent-512 | vector | 0.258 | 0.520 | 0.590 | 0.415 | 0.448 | 0.486 | 3742 |
| sent-512 | bm25 | 0.337 | 0.562 | 0.624 | 0.483 | 0.502 | 0.520 | 3712 |
| sent-512 | hybrid | 0.323 | 0.632 | 0.713 | 0.499 | 0.543 | 0.567 | 3766 |
| sent-512 | hybrid+rerank | 0.475 | 0.758 | 0.809 | 0.650 | 0.676 | 0.739 | 3787 |
| sent-512 | hybrid+graph | 0.211 | 0.626 | 0.739 | 0.438 | 0.504 | 0.584 | 3796 |
| sent-512 | hybrid+graph+rerank | 0.469 | 0.764 | 0.820 | 0.649 | 0.680 | 0.739 | 3791 |

**By question type (Recall@8, best stage per profile):**

| chunk profile | stage | single_hop | multi_hop |
|---|---|---|---|
| sent-512 | vector | 0.623 | 0.475 |
| sent-512 | bm25 | 0.652 | 0.525 |
| sent-512 | hybrid | 0.761 | 0.550 |
| sent-512 | hybrid+rerank | 0.862 | 0.625 |
| sent-512 | hybrid+graph | 0.768 | 0.637 |
| sent-512 | hybrid+graph+rerank | 0.862 | 0.675 |

**Latency (ms, p50 / p95):**

- `sent-512`: embed_query 41/193  vector_search 7/16  bm25_search 7/11  graph 15/40  rerank 2780/10467  (entity-link rate 99%)
