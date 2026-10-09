# LoCoMo results

Run dir: `cognee/eval_framework/locomo/report_artifacts`
Conversations: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## hybrid_completion_20_20_20

Questions with a corrupted gold answer left out (`without_corrupted_golds`): 143.

| questions | llm_judge |
| --- | ---: |
| total (open-domain left out) | 0.907 (n=1762) |

| question type | llm_judge |
| --- | ---: |
| single_hop | 0.918 (n=796) |
| multi_hop | 0.894 (n=236) |
| temporal | 0.908 (n=284) |
| open_domain | 0.593 (n=81) |
| adversarial | 0.892 (n=446) |

Run std (llm_judge): 0.000; unscored: 0; answer errors: 0.
