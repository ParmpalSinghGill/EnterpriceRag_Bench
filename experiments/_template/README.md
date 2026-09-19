# experiments/_template

Copy this folder for each new technique:

```bash
cp -r experiments/_template experiments/<technique-name>
```

## Hypothesis

What should this method improve (e.g. semantic Recall@10, multi-hop completeness, source routing)?

## Method

- Retrieval:
- Chunking:
- Rerank:
- Generation:
- Abstention / unanswerable handling:

## How to run

```bash
# example
python experiments/<technique-name>/run.py
```

## Outputs

- Answers: `answer_evaluation/<technique-name>/answers.jsonl`
- Metrics: `experiments/<technique-name>/outputs/metrics.json`

## Results (fill after eval)

| Metric | Value |
|---|---|
| Overall Score | |
| Correctness | |
| Completeness | |
| Document Recall | |
| Recall@10 | |
| Precision@10 | |
| MRR | |
| Hit@10 | |
| Source Accuracy | |
| Source Precision | |
| Source Recall | |

### Notable slices

- semantic:
- multi_hop:
- temporal:
- unanswerable:
