### End-to-end — `sherlock` (generator `MiniMax-M3.1-Flash`, judge `MiniMax-M3`, chunks `sent-512`)

| config | accuracy | single_hop | multi_hop | unanswerable_abstention | false_abstention | faithfulness | citation_precision | citation_recall | retrieval p50/p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| closed_book | 15.5% (10.1%–20.8%) | 16.1% | 12.9% | 45.0% | 6.0% | — | — | 0.0% | — |
| hybrid+rerank | 85.1% (79.8%–89.9%) | 91.2% | 58.1% | 80.0% | 5.4% | 94.0% | 97.2% | 93.8% | 2462/2600 |
| hybrid+rerank+graph | 86.3% (81.0%–91.1%) | 89.8% | 71.0% | 85.0% | 3.0% | 92.9% | 96.5% | 94.9% | 2491/2626 |
