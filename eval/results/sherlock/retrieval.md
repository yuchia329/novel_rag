### Retrieval — `sherlock` (168 answerable questions)

| chunk profile | stage | recall@1 | recall@5 | recall@8 | mrr@20 | ndcg@10 | recall@2000tok | ctx tok@8 |
|---|---|---|---|---|---|---|---|---|
| fixed-512 | vector | 0.455 | 0.741 | 0.804 | 0.608 | 0.640 | 0.643 | 4053 |
| fixed-512 | bm25 | 0.518 | 0.774 | 0.851 | 0.680 | 0.710 | 0.738 | 4020 |
| fixed-512 | hybrid | 0.568 | 0.798 | 0.848 | 0.712 | 0.736 | 0.750 | 4038 |
| fixed-512 | hybrid+rerank | 0.699 | 0.860 | 0.908 | 0.824 | 0.825 | 0.812 | 4056 |
| fixed-512 | hybrid+graph | 0.482 | 0.801 | 0.848 | 0.663 | 0.697 | 0.738 | 4039 |
| fixed-512 | hybrid+graph+rerank | 0.699 | 0.860 | 0.908 | 0.824 | 0.825 | 0.812 | 4056 |
| sent-256 | vector | 0.399 | 0.607 | 0.676 | 0.538 | 0.560 | 0.676 | 1912 |
| sent-256 | bm25 | 0.414 | 0.679 | 0.714 | 0.561 | 0.583 | 0.714 | 1901 |
| sent-256 | hybrid | 0.455 | 0.679 | 0.750 | 0.592 | 0.620 | 0.750 | 1911 |
| sent-256 | hybrid+rerank | 0.625 | 0.795 | 0.845 | 0.755 | 0.761 | 0.845 | 1916 |
| sent-256 | hybrid+graph | 0.449 | 0.679 | 0.750 | 0.589 | 0.618 | 0.750 | 1912 |
| sent-256 | hybrid+graph+rerank | 0.625 | 0.795 | 0.845 | 0.755 | 0.761 | 0.845 | 1916 |
| sent-512 | vector | 0.440 | 0.750 | 0.792 | 0.608 | 0.643 | 0.708 | 3895 |
| sent-512 | bm25 | 0.479 | 0.750 | 0.836 | 0.645 | 0.675 | 0.711 | 3832 |
| sent-512 | hybrid | 0.536 | 0.810 | 0.851 | 0.691 | 0.719 | 0.768 | 3870 |
| sent-512 | hybrid+rerank | 0.682 | 0.878 | 0.899 | 0.822 | 0.828 | 0.854 | 3929 |
| sent-512 | hybrid+graph | 0.500 | 0.804 | 0.848 | 0.670 | 0.704 | 0.756 | 3871 |
| sent-512 | hybrid+graph+rerank | 0.682 | 0.878 | 0.899 | 0.822 | 0.828 | 0.854 | 3929 |
| sent-1024 | vector | 0.366 | 0.753 | 0.836 | 0.575 | 0.636 | 0.429 | 7867 |
| sent-1024 | bm25 | 0.560 | 0.836 | 0.899 | 0.731 | 0.754 | 0.568 | 7800 |
| sent-1024 | hybrid | 0.536 | 0.842 | 0.908 | 0.723 | 0.756 | 0.586 | 7830 |
| sent-1024 | hybrid+rerank | 0.720 | 0.881 | 0.929 | 0.843 | 0.853 | 0.729 | 7991 |
| sent-1024 | hybrid+graph | 0.455 | 0.836 | 0.917 | 0.674 | 0.722 | 0.515 | 7830 |
| sent-1024 | hybrid+graph+rerank | 0.717 | 0.881 | 0.929 | 0.840 | 0.851 | 0.726 | 7991 |
| pc-128-768 | vector | 0.411 | 0.729 | 0.798 | 0.573 | 0.616 | 0.527 | 5919 |
| pc-128-768 | bm25 | 0.423 | 0.708 | 0.789 | 0.596 | 0.641 | 0.583 | 5885 |
| pc-128-768 | hybrid | 0.417 | 0.735 | 0.827 | 0.601 | 0.645 | 0.568 | 5879 |
| pc-128-768 | hybrid+rerank | 0.577 | 0.798 | 0.857 | 0.720 | 0.741 | 0.688 | 5989 |
| pc-128-768 | hybrid+graph | 0.423 | 0.735 | 0.827 | 0.601 | 0.644 | 0.560 | 5884 |
| pc-128-768 | hybrid+graph+rerank | 0.577 | 0.798 | 0.857 | 0.720 | 0.741 | 0.688 | 5989 |

**By question type (Recall@8, best stage per profile):**

| chunk profile | stage | single_hop | multi_hop |
|---|---|---|---|
| fixed-512 | vector | 0.869 | 0.516 |
| fixed-512 | bm25 | 0.912 | 0.581 |
| fixed-512 | hybrid | 0.905 | 0.597 |
| fixed-512 | hybrid+rerank | 0.964 | 0.661 |
| fixed-512 | hybrid+graph | 0.905 | 0.597 |
| fixed-512 | hybrid+graph+rerank | 0.964 | 0.661 |
| sent-256 | vector | 0.745 | 0.371 |
| sent-256 | bm25 | 0.766 | 0.484 |
| sent-256 | hybrid | 0.818 | 0.452 |
| sent-256 | hybrid+rerank | 0.920 | 0.516 |
| sent-256 | hybrid+graph | 0.818 | 0.452 |
| sent-256 | hybrid+graph+rerank | 0.920 | 0.516 |
| sent-512 | vector | 0.861 | 0.484 |
| sent-512 | bm25 | 0.891 | 0.597 |
| sent-512 | hybrid | 0.905 | 0.613 |
| sent-512 | hybrid+rerank | 0.942 | 0.710 |
| sent-512 | hybrid+graph | 0.898 | 0.629 |
| sent-512 | hybrid+graph+rerank | 0.942 | 0.710 |
| sent-1024 | vector | 0.876 | 0.661 |
| sent-1024 | bm25 | 0.927 | 0.774 |
| sent-1024 | hybrid | 0.949 | 0.726 |
| sent-1024 | hybrid+rerank | 0.956 | 0.806 |
| sent-1024 | hybrid+graph | 0.949 | 0.774 |
| sent-1024 | hybrid+graph+rerank | 0.956 | 0.806 |
| pc-128-768 | vector | 0.854 | 0.548 |
| pc-128-768 | bm25 | 0.847 | 0.532 |
| pc-128-768 | hybrid | 0.876 | 0.613 |
| pc-128-768 | hybrid+rerank | 0.912 | 0.613 |
| pc-128-768 | hybrid+graph | 0.876 | 0.613 |
| pc-128-768 | hybrid+graph+rerank | 0.912 | 0.613 |

**Latency (ms, p50 / p95):**

- `fixed-512`: embed_query 40/69  vector_search 6/7  bm25_search 6/6  graph 5/12  rerank 2444/2599  (entity-link rate 89%)
- `sent-256`: embed_query 18/41  vector_search 6/6  bm25_search 5/6  graph 5/11  rerank 1081/1171  (entity-link rate 89%)
- `sent-512`: embed_query 43/60  vector_search 6/7  bm25_search 6/6  graph 5/12  rerank 2424/2548  (entity-link rate 89%)
- `sent-1024`: embed_query 64/766  vector_search 8/18  bm25_search 8/17  graph 8/22  rerank 10101/21245  (entity-link rate 89%)
- `pc-128-768`: embed_query 208/307  vector_search 11/24  bm25_search 9/24  graph 8/28  rerank 2553/3173  (entity-link rate 89%)
