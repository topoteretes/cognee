# BROAD: large mock-corpus answer checks

These live-model results were measured before the default changed from 1 GB to 512 MB
and before store operations moved onto a worker. The 139.5 MB fixture fits either
threshold. The historical model responses are preserved below; the subsequent worker
validation is recorded in [the storage benchmark](broad_store_benchmark.md).

On this fixture, the draft's full-coverage path answered **7/7 scenarios correctly**;
the previous 32 MB path answered **3/7** against the same full-dataset ground truth.
The join passed after the existing SQL timeout/retry recovered from an inefficient
first query. This is a functional comparison on one synthetic corpus, not a general
retrieval-accuracy estimate.

## What ran

Measured on 2026-11-23 against production code at
`556604528c233cc6a5d7314e9c09a15cb811d7c4` in draft PR #6826.
The baseline store is from dev commit `e919461945cc8603e1084c7380c075b6449bae46`.

- **139,466,344 actual UTF-8 bytes**, across **929 documents**.
- **2,000,000 orders**, **100,000 customers**, **117,648 refunds** and
  **100,000 support notes**, plus one rollout update.
- Python generated every record and independently accumulated the expected answers.
  No LLM, embedding model or local model created the data.
- **GPT-6 Luna** (`openai/gpt-6-luna`) made the real BROAD decisions, wrote the SQL and
  phrased the answers. The old run made 14 gateway calls; the new run made 15,
  including the join retry. No expected answer or reference SQL was supplied to it.
- The real corpus loader, record parser, schema description, SQL execution, reading
  route, completion and store cleanup ran. The graph, vector and stored-size lookups
  were mocked. The dataset was never presented as a whole LLM prompt.
- The old configuration uses a file-backed store and `store_mb=32`; the candidate
  uses RAM and `store_mb=1000`. Prompts, query limits and dependencies are shared.

The vector mock ranks documents by lexical overlap with the question, then by document
name for ties; it uses no embeddings. Each document is one mock chunk. Consequently,
these results test the effect of coverage under a specified ranking and do not estimate
the quality of real vector rankings or production ingestion/chunking. Some mock chunks
are much larger than ordinary ingestion chunks; the largest holds 455,034 stored bytes.

## Results

All monetary amounts below are integer cents, as stored in the fixture.

| Question | Full-dataset truth / candidate result | Previous 32 MB result |
| --- | --- | --- |
| Paid-order count and sum | 1,000,000; 59,975,943,147 | 283,750; 14,214,238,125 |
| Paid-order counts and sums by region | All three groups exact; see below | All three undercounted |
| Highest-value order | O1999999; 9,876,543,210 | Maximum 100,099; missed the outlier |
| Paid orders for enterprise customers in Serbia | 166,680; 18,227,278,307 | Could not answer; required customer table absent from the selected store |
| Jose Alvarez customer count, ignoring case/accents | 30,000 | 30,000 |
| Distinct refunded orders and total refunds | 117,648; 3,942,563,747 | 117,648; 3,942,563,747 |
| Why Project Sequoia moved from May to June | Supplier failed the security audit | Same correct explanation |

| Region | Correct paid count | Correct amount_cents | Old paid count | Old amount_cents |
| --- | ---: | ---: | ---: | ---: |
| AMER | 333,332 | 16,699,766,558 | 94,582 | 4,737,980,933 |
| APAC | 333,334 | 26,576,276,584 | 94,584 | 4,738,177,812 |
| EMEA | 333,334 | 16,699,900,005 | 94,584 | 4,738,079,380 |

The old partial results carry the existing warning that the rest of the dataset was
not searched. A mismatch here means an answer differs from the question's full-dataset
truth; the selected subset's SQL result is not necessarily computed incorrectly.
The old name and refund questions happened to include all relevant rows, so their
answers already matched. Both rollout answers came from one selected passage; raising
the record-store threshold did not increase the semantic reading budget.

A separate control loaded the **complete corpus into the old disk-backed store** and
replayed the candidate's six generated numerical queries. Every result matched both
the RAM result and Python ground truth. Thus the observed accuracy improvement comes
from increasing coverage, rather than from RAM changing SQL results.

All six numerical checks compare the returned SQL rows with the independent Python
answers. The natural-language answers were also inspected. The explanation matches the
supplied rollout note; its automated keyword check is only a lightweight check of that
one scenario, not a semantic answer-quality metric.

## Performance issue exposed by the join

The initial generated query joined identifiers with:

```sql
ON norm(o."customer_id") = norm(c."customer_id")
```

It hit the existing **120-second SQL timeout**. BROAD's normal retry generated:

```sql
ON o."customer_id" = c."customer_id"
```

and returned the correct answer. These fixture identifiers already have identical
canonical spellings. `EXPLAIN QUERY PLAN` on the complete disk-store control showed:

| Join | Query plan |
| --- | --- |
| `norm(o.customer_id) = norm(c.customer_id)` | `SCAN c`; `SCAN o` |
| `o.customer_id = c.customer_id` | `SCAN c`; `SEARCH o USING AUTOMATIC PARTIAL COVERING INDEX (customer_id=?)` |

These are the observed plans on SQLite 3.53.1. SQLite documents the distinction between
[scanning a table and searching an index](https://www.sqlite.org/eqp.html), and its
[automatic query-time indexes](https://www.sqlite.org/optoverview.html#autoindex).
The first query's function-wrapped identifiers prevented that efficient access path
in this store; this is not a claim that expression joins can never be indexed.

The full question took **235.4955 seconds**, including loading,
schema preparation, the timeout, retry and answer. The other candidate scenarios took
**85.3942–109.7218 seconds**, with **71.30–94.41 seconds** in schema description.
This confirms that larger coverage is useful but does not make BROAD responsive.

The old and new runs overlapped in separate processes for part of the experiment.
These elapsed times are diagnostic observations, not a controlled speed comparison.
The earlier serial storage benchmark remains a separate experiment.

No production normalization, prompts, SQL limits, or worker execution changed during
these original live-model tests. The later worker change moves synchronous work off the
event loop. Reducing repeated query-hint normalization and guiding identifier joins
toward direct equality remain follow-up improvements to evaluate.

## Reproduce

The fixture contains synthetic business data only. Live mode requires a valid API key;
use `--env-file` with an existing local file containing `LLM_API_KEY`, or set that
environment variable. The harness selects GPT-6 Luna. Do not commit credentials.

```sh
uv run python cognee/tests/performance/benchmark_broad_quality.py --fixture-dir /tmp/broad-quality --generate-only
git show e919461945cc8603e1084c7380c075b6449bae46:cognee/modules/retrieval/broad_store.py > /tmp/broad-disk.py
PYTHONHASHSEED=0 uv run python cognee/tests/performance/benchmark_broad_quality.py --fixture-dir /tmp/broad-quality --variant old --disk-store-source /tmp/broad-disk.py --env-file /path/to/.env --json-output /tmp/broad-old-quality.json
PYTHONHASHSEED=0 uv run python cognee/tests/performance/benchmark_broad_quality.py --fixture-dir /tmp/broad-quality --variant new --env-file /path/to/.env --json-output /tmp/broad-new-quality.json
PYTHONHASHSEED=0 uv run python cognee/tests/performance/benchmark_broad_quality.py --fixture-dir /tmp/broad-quality --variant disk-full --disk-store-source /tmp/broad-disk.py --replay-sql /tmp/broad-new-quality.json --json-output /tmp/broad-disk-replay.json
```

`--fixed-sql` runs the six numerical cases without any model, using reference SQL.
It verifies execution and coverage, not natural-language query generation. SQL replay
likewise uses no model and skips query-hint preparation; it checks storage equivalence
and collects query plans, rather than measuring normal question latency.

The final mock ranks whole documents. Its rankings were checked against the live-run
prototype's prefix-based scoring: all observed vector-query rankings were identical
except for an unselected refund chunk in the paid-count case. A zero-model rerun of
that case confirmed identical retained table/line counts, coverage and numerical result.
This harness refinement did not change the information used for the reported answers.

The tests do not cover ingestion/extraction quality, authorization, session behavior,
dataset routing, live graph/vector adapters, arbitrary document formats, or datasets
above 1 GB. They do not establish that semantic questions can read the whole corpus.
