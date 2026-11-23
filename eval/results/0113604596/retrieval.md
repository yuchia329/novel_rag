### Retrieval — `0113604596` (178 answerable questions)

| chunk profile | stage | recall@1 | recall@5 | recall@8 | mrr@20 | ndcg@10 | recall@2000tok | ctx tok@8 |
|---|---|---|---|---|---|---|---|---|
| fixed-512 | vector | 0.264 | 0.497 | 0.565 | 0.419 | 0.441 | 0.455 | 3845 |
| fixed-512 | bm25 | 0.329 | 0.570 | 0.612 | 0.481 | 0.504 | 0.520 | 3807 |
| fixed-512 | hybrid | 0.343 | 0.618 | 0.719 | 0.506 | 0.549 | 0.553 | 3873 |
| fixed-512 | hybrid+rerank | 0.506 | 0.747 | 0.795 | 0.667 | 0.682 | 0.719 | 3878 |
| fixed-512 | hybrid+graph | 0.242 | 0.638 | 0.736 | 0.444 | 0.507 | 0.522 | 3919 |
| fixed-512 | hybrid+graph+rerank | 0.500 | 0.756 | 0.806 | 0.666 | 0.685 | 0.725 | 3884 |
| sent-256 | vector | 0.289 | 0.565 | 0.598 | 0.445 | 0.477 | 0.604 | 1880 |
| sent-256 | bm25 | 0.289 | 0.514 | 0.576 | 0.432 | 0.452 | 0.579 | 1886 |
| sent-256 | hybrid | 0.346 | 0.596 | 0.669 | 0.500 | 0.525 | 0.674 | 1896 |
| sent-256 | hybrid+rerank | 0.500 | 0.699 | 0.744 | 0.639 | 0.644 | 0.744 | 1907 |
| sent-256 | hybrid+graph | 0.326 | 0.598 | 0.666 | 0.492 | 0.525 | 0.666 | 1894 |
| sent-256 | hybrid+graph+rerank | 0.500 | 0.702 | 0.747 | 0.640 | 0.647 | 0.747 | 1907 |
| sent-512 | vector | 0.258 | 0.520 | 0.590 | 0.415 | 0.448 | 0.486 | 3742 |
| sent-512 | bm25 | 0.337 | 0.562 | 0.624 | 0.483 | 0.502 | 0.520 | 3712 |
| sent-512 | hybrid | 0.323 | 0.632 | 0.713 | 0.499 | 0.543 | 0.567 | 3766 |
| sent-512 | hybrid+rerank | 0.475 | 0.758 | 0.809 | 0.650 | 0.676 | 0.739 | 3787 |
| sent-512 | hybrid+graph | 0.216 | 0.638 | 0.733 | 0.434 | 0.503 | 0.573 | 3796 |
| sent-512 | hybrid+graph+rerank | 0.469 | 0.764 | 0.820 | 0.650 | 0.680 | 0.744 | 3793 |
| sent-1024 | vector | 0.326 | 0.567 | 0.624 | 0.464 | 0.490 | 0.379 | 7226 |
| sent-1024 | bm25 | 0.331 | 0.615 | 0.680 | 0.502 | 0.533 | 0.455 | 7174 |
| sent-1024 | hybrid | 0.402 | 0.685 | 0.739 | 0.569 | 0.603 | 0.494 | 7218 |
| sent-1024 | hybrid+rerank | 0.478 | 0.792 | 0.848 | 0.665 | 0.700 | 0.584 | 7453 |
| sent-1024 | hybrid+graph | 0.309 | 0.638 | 0.750 | 0.502 | 0.549 | 0.357 | 7421 |
| sent-1024 | hybrid+graph+rerank | 0.478 | 0.801 | 0.860 | 0.670 | 0.705 | 0.590 | 7483 |
| pc-128-768 | vector | 0.343 | 0.610 | 0.677 | 0.497 | 0.532 | 0.458 | 5756 |
| pc-128-768 | bm25 | 0.275 | 0.500 | 0.570 | 0.417 | 0.441 | 0.374 | 5753 |
| pc-128-768 | hybrid | 0.362 | 0.618 | 0.705 | 0.523 | 0.560 | 0.497 | 5755 |
| pc-128-768 | hybrid+rerank | 0.452 | 0.708 | 0.753 | 0.611 | 0.634 | 0.579 | 5759 |
| pc-128-768 | hybrid+graph | 0.351 | 0.610 | 0.677 | 0.515 | 0.548 | 0.497 | 5794 |
| pc-128-768 | hybrid+graph+rerank | 0.452 | 0.708 | 0.753 | 0.611 | 0.634 | 0.579 | 5762 |

**By question type (Recall@8, best stage per profile):**

| chunk profile | stage | single_hop | multi_hop |
|---|---|---|---|
| fixed-512 | vector | 0.587 | 0.487 |
| fixed-512 | bm25 | 0.638 | 0.525 |
| fixed-512 | hybrid | 0.768 | 0.550 |
| fixed-512 | hybrid+rerank | 0.848 | 0.613 |
| fixed-512 | hybrid+graph | 0.754 | 0.675 |
| fixed-512 | hybrid+graph+rerank | 0.848 | 0.662 |
| sent-256 | vector | 0.645 | 0.438 |
| sent-256 | bm25 | 0.609 | 0.463 |
| sent-256 | hybrid | 0.725 | 0.475 |
| sent-256 | hybrid+rerank | 0.804 | 0.537 |
| sent-256 | hybrid+graph | 0.696 | 0.562 |
| sent-256 | hybrid+graph+rerank | 0.804 | 0.550 |
| sent-512 | vector | 0.623 | 0.475 |
| sent-512 | bm25 | 0.652 | 0.525 |
| sent-512 | hybrid | 0.761 | 0.550 |
| sent-512 | hybrid+rerank | 0.862 | 0.625 |
| sent-512 | hybrid+graph | 0.768 | 0.613 |
| sent-512 | hybrid+graph+rerank | 0.862 | 0.675 |
| sent-1024 | vector | 0.652 | 0.525 |
| sent-1024 | bm25 | 0.710 | 0.575 |
| sent-1024 | hybrid | 0.768 | 0.637 |
| sent-1024 | hybrid+rerank | 0.884 | 0.725 |
| sent-1024 | hybrid+graph | 0.768 | 0.688 |
| sent-1024 | hybrid+graph+rerank | 0.891 | 0.750 |
| pc-128-768 | vector | 0.710 | 0.562 |
| pc-128-768 | bm25 | 0.594 | 0.487 |
| pc-128-768 | hybrid | 0.732 | 0.613 |
| pc-128-768 | hybrid+rerank | 0.775 | 0.675 |
| pc-128-768 | hybrid+graph | 0.710 | 0.562 |
| pc-128-768 | hybrid+graph+rerank | 0.775 | 0.675 |

**Latency (ms, p50 / p95):**

- `fixed-512`: embed_query 480/718  vector_search 13/23  bm25_search 10/20  graph 24/51  rerank 5230/6688  (entity-link rate 99%)
- `sent-256`: embed_query 175/674  vector_search 13/29  bm25_search 11/31  graph 27/84  rerank 2490/3699  (entity-link rate 99%)
- `sent-512`: embed_query 211/680  vector_search 16/27  bm25_search 11/21  graph 25/63  rerank 11132/14618  (entity-link rate 99%)
- `sent-1024`: embed_query 50/450  vector_search 8/17  bm25_search 7/12  graph 17/24  rerank 6825/23491  (entity-link rate 99%)
- `pc-128-768`: embed_query 17/24  vector_search 8/9  bm25_search 6/7  graph 17/32  rerank 587/708  (entity-link rate 99%)
