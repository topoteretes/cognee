"""Compare BROAD coverage against independently generated business-data answers.

Data generation uses Python only. Live mode uses the configured LLM for the actual
BROAD decision and answer; --fixed-sql tests execution/coverage without any model.
The mock vector ranks chunks by lexical overlap, not embeddings. This measures the
retriever, excluding ingestion, permissions, routing, sessions and real DB adapters.
"""

import argparse
import asyncio
import csv
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

ORDERS = 2_000_000
CUSTOMERS = 100_000
NOTES = 100_000
NAMES = (
    "José Álvarez",
    "JOSE ALVAREZ",
    "Jose\u0301 A\u0301lvarez",
    "Ana María",
    "İpek Öztürk",
    "Milan Petrović",
    "Emilie Dubois",
    "Aiko Tanaka",
    "Lucas Silva",
    "Sara Novak",
)
COUNTRIES = ("Germany", "Japan", "Brazil", "Serbia")
SEGMENTS = ("enterprise", "business", "personal")
REGIONS = ("EMEA", "APAC", "AMER")
PRODUCTS = ("subscription", "hardware", "training")
STATUSES = ("paid", "pending", "cancelled", "paid")
REASONS = ("duplicate billing", "customer cancellation", "delivery issue")

CASES = {
    "count": (
        (
            "How many orders have status paid, and what is their total amount_cents? "
            "Return the count and sum."
        ),
        "SELECT COUNT(*), SUM(amount_cents) FROM orders_0000 WHERE status = 'paid'",
    ),
    "regions": (
        (
            "For each region, count paid orders and sum amount_cents. "
            "Return region, count, and sum, sorted by region."
        ),
        (
            "SELECT region, COUNT(*), SUM(amount_cents) FROM orders_0000 "
            "WHERE status = 'paid' GROUP BY region ORDER BY region"
        ),
    ),
    "outlier": (
        "Which order has the highest amount_cents? Return only order_id and amount_cents.",
        "SELECT order_id, amount_cents FROM orders_0000 ORDER BY amount_cents DESC LIMIT 1",
    ),
    "join": (
        (
            "For paid orders placed by enterprise customers in Serbia, "
            "return the order count and total amount_cents."
        ),
        (
            "SELECT COUNT(*), SUM(o.amount_cents) FROM orders_0000 o "
            "JOIN customers_0000 c ON o.customer_id = c.customer_id "
            "WHERE o.status = 'paid' AND c.segment = 'enterprise' AND c.country = 'Serbia'"
        ),
    ),
    "unicode": (
        (
            "How many customers are named Jose Alvarez, ignoring accents and letter case? "
            "Return only the count."
        ),
        "SELECT COUNT(*) FROM customers_0000 WHERE norm(customer_name) = norm('Jose Alvarez')",
    ),
    "refunds": (
        "Across all refunds, return the count of distinct order_ids and total refund_cents.",
        "SELECT COUNT(DISTINCT order_id), SUM(refund_cents) FROM refunds_0000",
    ),
    "meaning": (
        "Why did Project Sequoia's rollout move from May to June?",
        None,
    ),
}


def generate(directory: Path):
    """Write actual UTF-8 files and compute truth with Python, independently of SQL."""
    directory.mkdir(parents=True, exist_ok=True)
    documents = []
    totals = defaultdict(int)
    regions = {region: [0, 0] for region in REGIONS}

    def save(name, text):
        path = directory / name
        data = text.encode("utf-8")
        path.write_bytes(data)
        documents.append(
            {"id": str(UUID(int=len(documents) + 1)), "name": name, "bytes": len(data)}
        )

    def partition(prefix, columns, rows):
        rows = iter(rows)
        index = 0
        while batch := list(islice(rows, 2500)):
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(batch)
            save(f"{prefix}_{index:04d}.csv", buffer.getvalue())
            index += 1

    def customers():
        for i in range(CUSTOMERS):
            yield (f"C{i:06d}", NAMES[i % len(NAMES)], COUNTRIES[i % 4], SEGMENTS[i % 3])

    def orders():
        for i in range(ORDERS):
            customer = i % CUSTOMERS
            amount = 9_876_543_210 if i == ORDERS - 1 else (i * 37) % 100_000 + 100
            status, region = STATUSES[i % 4], REGIONS[i % 3]
            if status == "paid":
                totals["count"] += 1
                totals["sum"] += amount
                regions[region][0] += 1
                regions[region][1] += amount
                if customer % 3 == 0 and customer % 4 == 3:
                    totals["join_count"] += 1
                    totals["join_sum"] += amount
            yield (
                f"O{i:07d}",
                f"C{customer:06d}",
                f"2026-{i % 12 + 1:02d}-{i % 28 + 1:02d}",
                status,
                amount,
                region,
                PRODUCTS[(i // 3) % 3],
            )

    def refunds():
        for i in range(0, ORDERS, 17):
            amount = 9_876_543_210 if i == ORDERS - 1 else (i * 37) % 100_000 + 100
            refund = amount // 4
            totals["refund_count"] += 1
            totals["refund_sum"] += refund
            yield (f"R{i:07d}", f"O{i:07d}", refund, REASONS[(i // 17) % 3])

    partition("customers", ("customer_id", "customer_name", "country", "segment"), customers())
    partition(
        "orders",
        ("order_id", "customer_id", "ordered_at", "status", "amount_cents", "region", "product"),
        orders(),
    )
    partition("refunds", ("refund_id", "order_id", "refund_cents", "reason"), refunds())
    for start in range(0, NOTES, 2500):
        text = "\n".join(
            f"Ticket {i:07d} concerns {PRODUCTS[i % 3]} renewal in {COUNTRIES[i % 4]} "
            f"and was resolved after the account manager verified invoice {i * 19:09d} "
            f"with the customer during the scheduled follow up on day {i % 28 + 1}."
            for i in range(start, start + 2500)
        )
        save(f"notes_{start // 2500:04d}.txt", text)
    save(
        "notes_9999_sequoia.txt",
        "Project Sequoia rollout update\nThe rollout moved from May to June because "
        "the supplier failed the security audit. Maya Chen owns the launch schedule.",
    )
    expected = {
        "count": [[totals["count"], totals["sum"]]],
        "regions": [[region, *regions[region]] for region in sorted(regions)],
        "outlier": [[f"O{ORDERS - 1:07d}", 9_876_543_210]],
        "join": [[totals["join_count"], totals["join_sum"]]],
        "unicode": [[sum(i % len(NAMES) in (0, 1, 2) for i in range(CUSTOMERS))]],
        "refunds": [[totals["refund_count"], totals["refund_sum"]]],
        "meaning": "The supplier failed the security audit.",
    }
    manifest = {
        "bytes": sum(doc["bytes"] for doc in documents),
        "orders": ORDERS,
        "customers": CUSTOMERS,
        "refunds": totals["refund_count"],
        "support_notes": NOTES,
        "documents": documents,
        "expected": expected,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "FIXTURE:", json.dumps({k: v for k, v in manifest.items() if k != "documents"}), flush=True
    )


def terms(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    return {word.removesuffix("s") for word in re.findall(r"[^\W_]+", folded)}


class Fixture:
    def __init__(self, directory):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.documents = {doc["id"]: doc for doc in self.manifest["documents"]}
        self.signatures = {
            doc["id"]: terms(doc["name"] + " " + self.text(doc)) for doc in self.documents.values()
        }
        self.graph_documents = set()
        self.vector_searches = []

    def text(self, document):
        return (self.directory / document["name"]).read_text()

    async def get_filtered_graph_data(self, filters):
        if filters[0]["type"] == ["SchemaTable"]:
            return [], []
        return [
            (d["id"], {"type": "TextDocument", "name": d["name"]}) for d in self.documents.values()
        ], []

    async def get_neighborhood(self, documents, depth=1):
        await asyncio.sleep(0)
        nodes, edges = [], []
        for identifier in documents:
            self.graph_documents.add(identifier)
            chunk = f"chunk-{identifier}"
            nodes.append(
                (
                    chunk,
                    {
                        "type": "DocumentChunk",
                        "chunk_index": 0,
                        "text": self.text(self.documents[identifier]),
                    },
                )
            )
            edges.append((chunk, identifier, "is_part_of", {}))
        return nodes, edges

    async def search(self, collection, query_text, query_vector=None, limit=None, **kwargs):
        self.vector_searches.append({"collection": collection, "query": query_text, "limit": limit})
        if collection != "DocumentChunk_text":
            return []
        words = terms(query_text)
        ranked = sorted(
            self.documents.values(),
            key=lambda doc: (-len(words & self.signatures[doc["id"]]), doc["name"]),
        )[:limit]
        return [
            SimpleNamespace(
                id=f"chunk-{d['id']}",
                score=-len(words & self.signatures[d["id"]]),
                payload=self.payload(d) if kwargs.get("include_payload") else None,
            )
            for d in ranked
        ]

    def payload(self, document):
        return {
            "text": self.text(document),
            "document_id": document["id"],
            "document_name": document["name"],
            "chunk_index": 0,
        }

    async def retrieve(self, collection, ids):
        documents = [self.documents[point_id.removeprefix("chunk-")] for point_id in ids]
        return [
            SimpleNamespace(id=f"chunk-{d['id']}", score=0, payload=self.payload(d))
            for d in documents
        ]


async def run(args):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.retrieval import broad_retriever
    from cognee.modules.retrieval.broad_retriever import (
        BroadLimitError,
        BroadNoTextError,
        BroadQueryError,
        BroadRetriever,
        DecisionLarge,
        DecisionSmall,
    )

    parent_store = broad_retriever.RecordStore
    if args.variant != "new":
        spec = importlib.util.spec_from_file_location("quality_disk_store", args.disk_store_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # The shared retriever must recognize baseline errors for its normal retry.
        module.BroadQueryError = BroadQueryError
        parent_store = module.RecordStore
    fixture = Fixture(args.fixture_dir)
    results = {
        "variant": args.variant,
        "mode": "fixed_sql" if args.fixed_sql else "live_llm",
        "model": None if args.fixed_sql else os.environ["LLM_MODEL"],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_bytes": fixture.manifest["bytes"],
        "documents": len(fixture.documents),
        "orders": ORDERS,
        "customers": CUSTOMERS,
        "refunds": fixture.manifest["refunds"],
        "support_notes": NOTES,
        "cases": [],
    }
    original_llm = LLMGateway.acreate_structured_output
    for identifier in args.scenarios or CASES:
        question, reference_sql = CASES[identifier]
        if args.fixed_sql and reference_sql is None:
            continue
        fixture.graph_documents.clear()
        fixture.vector_searches.clear()
        stage = defaultdict(float)
        storage, decisions = {}, []
        model_calls = 0

        class Store(parent_store):
            metrics = stage
            stats = storage

            def add_document(self, *values):
                started = time.perf_counter()
                super().add_document(*values)
                self.metrics["parse_insert"] += time.perf_counter() - started

            def describe(self, question):
                started = time.perf_counter()
                result = super().describe(question)
                self.metrics["describe"] += time.perf_counter() - started
                return result

            def close(self):
                self.stats["table_rows"] = {name: table.rows for name, table in self.tables.items()}
                self.stats["line_rows"] = self.line_rows
                self.stats["database_files"] = self.connection.execute(
                    "PRAGMA database_list"
                ).fetchall()
                super().close()

        class Retriever(BroadRetriever):
            metrics = stage
            responses = decisions
            fixed_sql = reference_sql

            async def document_sizes(self, documents):
                return {doc: fixture.documents[doc]["bytes"] for doc in documents}

            async def _ask(self, text_input, prompt, response_model):
                started = time.perf_counter()
                if args.fixed_sql:
                    assert response_model in (DecisionLarge, DecisionSmall)
                    response = response_model(sql=self.fixed_sql)
                else:
                    response = await super()._ask(text_input, prompt, response_model)
                self.metrics["decision_llm"] += time.perf_counter() - started
                self.responses.append({"prompt": prompt, "response": response.model_dump()})
                return response

        async def unified():
            return SimpleNamespace(graph=fixture, vector=fixture)

        async def llm(*values, **kwargs):
            nonlocal model_calls
            assert not args.fixed_sql, "Fixed-SQL mode must not call a model"
            model_calls += 1
            return await original_llm(*values, **kwargs)

        config = {"store_mb": 32} if args.variant == "old" else {}
        retriever = Retriever(system_prompt_path="broad_answer.txt", **config)
        row = {
            "id": identifier,
            "question": question,
            "expected": fixture.manifest["expected"][identifier],
        }
        started = time.perf_counter()
        print(f"START {args.variant} {identifier}", flush=True)
        try:
            with (
                patch.object(broad_retriever, "RecordStore", Store),
                patch.object(broad_retriever, "get_unified_engine", unified),
                patch.object(LLMGateway, "acreate_structured_output", llm),
            ):
                found = await retriever.get_retrieved_objects(question)
                row.update(
                    route=found.route,
                    sql=found.sql,
                    rows=found.rows,
                    coverage=found.coverage,
                    note=found.note,
                )
                if not args.fixed_sql:
                    context = await retriever.get_context_from_objects(question, found)
                    row["answers"] = await retriever.get_completion_from_context(
                        question, found, context=context
                    )
            if identifier == "meaning":
                text = " ".join(row["answers"]).casefold()
                row["matches_truth"] = "security" in text and "audit" in text
            else:
                row["matches_truth"] = [list(values) for values in found.rows] == row["expected"]
        except (BroadQueryError, BroadLimitError, BroadNoTextError) as error:
            message = str(error)
            secret = os.environ.get("LLM_API_KEY")
            row.update(
                error=type(error).__name__,
                error_message=message.replace(secret, "<redacted>") if secret else message,
                matches_truth=False,
            )
        row.update(
            seconds=round(time.perf_counter() - started, 4),
            stages=dict(stage),
            model_calls=model_calls,
            decisions=decisions,
            storage=storage,
            graph_documents_loaded=len(fixture.graph_documents),
            vector_searches=list(fixture.vector_searches),
        )
        results["cases"].append(row)
        args.json_output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print("RESULT:", json.dumps(row, ensure_ascii=False), flush=True)


async def replay(args):
    """Replay live SQL against a complete disk store, isolating storage correctness."""
    from cognee.modules.retrieval.broad_retriever import BroadRetriever, Corpus
    from cognee.modules.retrieval.broad_worker import BroadWorker

    fixture = Fixture(args.fixture_dir)
    spec = importlib.util.spec_from_file_location("quality_disk_store", args.disk_store_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = {"mode": "sql_replay", "variant": "disk-full", "model_calls": 0, "cases": []}
    async with BroadWorker(module.RecordStore) as worker:
        store = worker.store
        corpus = Corpus(
            graph=fixture,
            documents=sorted(fixture.documents, key=lambda key: fixture.documents[key]["name"]),
            names={key: doc["name"] for key, doc in fixture.documents.items()},
            sizes={key: doc["bytes"] for key, doc in fixture.documents.items()},
            batch_chars=256_000_000,
            worker=worker,
        )
        started = time.perf_counter()
        _, characters = await BroadRetriever().build(worker, corpus)
        results.update(
            build_seconds=time.perf_counter() - started,
            dataset_bytes=fixture.manifest["bytes"],
            characters=characters,
            graph_documents_loaded=len(fixture.graph_documents),
            table_rows={name: built.rows for name, built in store.tables.items()},
            line_rows=store.line_rows,
        )
        for case in json.loads(args.replay_sql.read_text())["cases"]:
            if not case.get("sql") or case["id"] == "meaning":
                continue
            started = time.perf_counter()
            columns, rows, truncated = await worker.run(store.run, case["sql"])
            plans = [
                {
                    "sql": attempt["response"]["sql"],
                    "plan": await worker.run(
                        lambda sql: store.connection.execute(
                            "EXPLAIN QUERY PLAN " + sql
                        ).fetchall(),
                        attempt["response"]["sql"],
                    ),
                }
                for attempt in case["decisions"]
                if attempt["response"].get("sql")
            ]
            results["cases"].append(
                {
                    "id": case["id"],
                    "sql": case["sql"],
                    "columns": columns,
                    "rows": rows,
                    "matches_ram": [list(values) for values in rows] == case["rows"],
                    "matches_truth": [list(values) for values in rows] == case["expected"],
                    "truncated": truncated,
                    "seconds": time.perf_counter() - started,
                    "decision_query_plans": plans,
                }
            )
    args.json_output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--variant", choices=("old", "new", "disk-full"), default="new")
    parser.add_argument("--disk-store-source", type=Path)
    parser.add_argument("--fixed-sql", action="store_true")
    parser.add_argument("--replay-sql", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(CASES))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.generate_only:
        generate(args.fixture_dir)
        return
    if args.json_output is None:
        parser.error("--json-output is required for runs")
    if args.variant != "new" and args.disk_store_source is None:
        parser.error("--disk-store-source is required for disk variants")
    if args.replay_sql:
        if args.variant != "disk-full":
            parser.error("--replay-sql requires --variant disk-full")
        args.fixed_sql = True
    if not args.fixed_sql:
        if args.env_file:
            from dotenv import dotenv_values

            key = dotenv_values(args.env_file).get("LLM_API_KEY")
            if not key:
                parser.error("No LLM_API_KEY in the supplied env file")
            os.environ["LLM_API_KEY"] = key
        if not os.environ.get("LLM_API_KEY"):
            parser.error("Live mode requires an LLM_API_KEY")
        os.environ["LLM_MODEL"] = "openai/gpt-6-luna"
        os.environ["LLM_PROVIDER"] = "openai"
    os.environ.setdefault("TELEMETRY_DISABLED", "1")
    os.environ.setdefault("ENV", "dev")
    asyncio.run(replay(args) if args.replay_sql else run(args))


if __name__ == "__main__":
    main()
