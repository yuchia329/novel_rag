### Retrieval — `xiyouji` (168 answerable questions)

| chunk profile | stage | recall@1 | recall@5 | recall@8 | mrr@20 | ndcg@10 | recall@2000tok | ctx tok@8 |
|---|---|---|---|---|---|---|---|---|
| fixed-512 | vector | 0.318 | 0.568 | 0.652 | 0.453 | 0.495 | 0.482 | 4014 |
| fixed-512 | bm25 | 0.369 | 0.595 | 0.640 | 0.509 | 0.532 | 0.542 | 4017 |
| fixed-512 | hybrid | 0.381 | 0.670 | 0.738 | 0.548 | 0.582 | 0.571 | 4018 |
| fixed-512 | hybrid+rerank | 0.595 | 0.792 | 0.821 | 0.724 | 0.735 | 0.747 | 4030 |
| fixed-512 | hybrid+graph | 0.345 | 0.673 | 0.747 | 0.530 | 0.571 | 0.557 | 4022 |
| fixed-512 | hybrid+graph+rerank | 0.595 | 0.804 | 0.833 | 0.728 | 0.741 | 0.753 | 4031 |
| sent-256 | vector | 0.307 | 0.598 | 0.637 | 0.445 | 0.485 | 0.637 | 1940 |
| sent-256 | bm25 | 0.259 | 0.443 | 0.488 | 0.365 | 0.390 | 0.482 | 1926 |
| sent-256 | hybrid | 0.351 | 0.607 | 0.661 | 0.481 | 0.519 | 0.661 | 1933 |
| sent-256 | hybrid+rerank | 0.521 | 0.696 | 0.744 | 0.625 | 0.643 | 0.744 | 1941 |
| sent-256 | hybrid+graph | 0.339 | 0.601 | 0.661 | 0.474 | 0.510 | 0.661 | 1935 |
| sent-256 | hybrid+graph+rerank | 0.521 | 0.696 | 0.744 | 0.625 | 0.643 | 0.744 | 1941 |
| sent-512 | vector | 0.298 | 0.518 | 0.580 | 0.424 | 0.462 | 0.473 | 3925 |
| sent-512 | bm25 | 0.354 | 0.562 | 0.604 | 0.478 | 0.506 | 0.527 | 3926 |
| sent-512 | hybrid | 0.327 | 0.607 | 0.702 | 0.497 | 0.541 | 0.551 | 3933 |
| sent-512 | hybrid+rerank | 0.574 | 0.783 | 0.818 | 0.701 | 0.718 | 0.747 | 3937 |
| sent-512 | hybrid+graph | 0.298 | 0.616 | 0.714 | 0.480 | 0.528 | 0.548 | 3935 |
| sent-512 | hybrid+graph+rerank | 0.580 | 0.789 | 0.830 | 0.708 | 0.726 | 0.753 | 3937 |
| sent-1024 | vector | 0.232 | 0.521 | 0.622 | 0.397 | 0.450 | 0.253 | 7798 |
| sent-1024 | bm25 | 0.399 | 0.652 | 0.729 | 0.548 | 0.584 | 0.438 | 7770 |
| sent-1024 | hybrid | 0.363 | 0.702 | 0.762 | 0.545 | 0.598 | 0.387 | 7809 |
| sent-1024 | hybrid+rerank | 0.542 | 0.804 | 0.851 | 0.689 | 0.721 | 0.565 | 7927 |
| sent-1024 | hybrid+graph | 0.336 | 0.688 | 0.762 | 0.530 | 0.583 | 0.354 | 7868 |
| sent-1024 | hybrid+graph+rerank | 0.542 | 0.801 | 0.854 | 0.689 | 0.724 | 0.565 | 7931 |
| pc-128-768 | vector | 0.372 | 0.574 | 0.643 | 0.491 | 0.514 | 0.467 | 5936 |
| pc-128-768 | bm25 | 0.262 | 0.476 | 0.542 | 0.391 | 0.415 | 0.360 | 5935 |
| pc-128-768 | hybrid | 0.357 | 0.580 | 0.655 | 0.495 | 0.527 | 0.482 | 5940 |
| pc-128-768 | hybrid+rerank | 0.497 | 0.696 | 0.750 | 0.637 | 0.647 | 0.607 | 5956 |
| pc-128-768 | hybrid+graph | 0.357 | 0.580 | 0.655 | 0.495 | 0.526 | 0.482 | 5940 |
| pc-128-768 | hybrid+graph+rerank | 0.497 | 0.696 | 0.750 | 0.637 | 0.647 | 0.607 | 5956 |

**By question type (Recall@8, best stage per profile):**

| chunk profile | stage | single_hop | multi_hop |
|---|---|---|---|
| fixed-512 | vector | 0.691 | 0.484 |
| fixed-512 | bm25 | 0.676 | 0.484 |
| fixed-512 | hybrid | 0.772 | 0.594 |
| fixed-512 | hybrid+rerank | 0.853 | 0.688 |
| fixed-512 | hybrid+graph | 0.787 | 0.578 |
| fixed-512 | hybrid+graph+rerank | 0.868 | 0.688 |
| sent-256 | vector | 0.721 | 0.281 |
| sent-256 | bm25 | 0.529 | 0.312 |
| sent-256 | hybrid | 0.721 | 0.406 |
| sent-256 | hybrid+rerank | 0.801 | 0.500 |
| sent-256 | hybrid+graph | 0.721 | 0.406 |
| sent-256 | hybrid+graph+rerank | 0.801 | 0.500 |
| sent-512 | vector | 0.618 | 0.422 |
| sent-512 | bm25 | 0.640 | 0.453 |
| sent-512 | hybrid | 0.735 | 0.562 |
| sent-512 | hybrid+rerank | 0.860 | 0.641 |
| sent-512 | hybrid+graph | 0.750 | 0.562 |
| sent-512 | hybrid+graph+rerank | 0.875 | 0.641 |
| sent-1024 | vector | 0.654 | 0.484 |
| sent-1024 | bm25 | 0.765 | 0.578 |
| sent-1024 | hybrid | 0.809 | 0.562 |
| sent-1024 | hybrid+rerank | 0.882 | 0.719 |
| sent-1024 | hybrid+graph | 0.809 | 0.562 |
| sent-1024 | hybrid+graph+rerank | 0.882 | 0.734 |
| pc-128-768 | vector | 0.691 | 0.438 |
| pc-128-768 | bm25 | 0.566 | 0.438 |
| pc-128-768 | hybrid | 0.691 | 0.500 |
| pc-128-768 | hybrid+rerank | 0.787 | 0.594 |
| pc-128-768 | hybrid+graph | 0.691 | 0.500 |
| pc-128-768 | hybrid+graph+rerank | 0.787 | 0.594 |

**Latency (ms, p50 / p95):**

- `fixed-512`: embed_query 459/870  vector_search 11/21  bm25_search 10/20  graph 11/39  rerank 4780/5721  (entity-link rate 99%)
- `sent-256`: embed_query 64/383  vector_search 8/19  bm25_search 8/17  graph 9/24  rerank 1211/2567  (entity-link rate 99%)
- `sent-512`: embed_query 255/844  vector_search 10/22  bm25_search 12/34  graph 9/41  rerank 4011/5667  (entity-link rate 99%)
- `sent-1024`: embed_query 231/747  vector_search 13/22  bm25_search 10/18  graph 11/41  rerank 18982/24051  (entity-link rate 99%)
- `pc-128-768`: embed_query 117/160  vector_search 13/15  bm25_search 8/11  graph 8/22  rerank 1164/1297  (entity-link rate 99%)
