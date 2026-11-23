### Retrieval — `sherlock` (168 answerable questions)

| chunk profile | stage | recall@1 | recall@5 | recall@8 | mrr@20 | ndcg@10 | recall@2000tok | ctx tok@8 |
|---|---|---|---|---|---|---|---|---|
| sent-512 | vector | 0.440 | 0.750 | 0.792 | 0.608 | 0.643 | 0.708 | 3895 |
| sent-512 | bm25 | 0.479 | 0.750 | 0.836 | 0.645 | 0.675 | 0.711 | 3832 |
| sent-512 | hybrid | 0.536 | 0.810 | 0.851 | 0.691 | 0.719 | 0.768 | 3870 |
| sent-512 | hybrid+rerank | 0.682 | 0.878 | 0.899 | 0.822 | 0.828 | 0.854 | 3929 |
| sent-512 | hybrid+graph | 0.494 | 0.804 | 0.848 | 0.665 | 0.700 | 0.756 | 3872 |
| sent-512 | hybrid+graph+rerank | 0.682 | 0.878 | 0.899 | 0.822 | 0.828 | 0.854 | 3929 |

**By question type (Recall@8, best stage per profile):**

| chunk profile | stage | single_hop | multi_hop |
|---|---|---|---|
| sent-512 | vector | 0.861 | 0.484 |
| sent-512 | bm25 | 0.891 | 0.597 |
| sent-512 | hybrid | 0.905 | 0.613 |
| sent-512 | hybrid+rerank | 0.942 | 0.710 |
| sent-512 | hybrid+graph | 0.898 | 0.629 |
| sent-512 | hybrid+graph+rerank | 0.942 | 0.710 |

**Latency (ms, p50 / p95):**

- `sent-512`: embed_query 178/579  vector_search 11/30  bm25_search 8/26  graph 11/35  rerank 9073/10767  (entity-link rate 89%)
