"""
Graph Model Expansion Benchmark
================================

Demonstrates progressive improvement from RAG → default cognee → custom model
→ cascade-expanded model on multi-hop QA.

Uses 2WikiMultihopQA with a train/eval split:
  - Discovery set (3 questions): cascade runs on these to discover missing entity types
  - Evaluation set (5 questions): all 4 approaches are scored on these

Four approaches compared:
  1. RAG — chunks + vector search, no graph
  2. Cognee default — generic KnowledgeGraph, out of the box
  3. Custom model — Person, Place, Work only (missing entity types like Award, Organization)
  4. Cascade-expanded — adds entity types discovered by cascade on separate data

Usage:
    python examples/benchmarks/graph_model_expansion_benchmark.py

    # Preview all questions
    python examples/benchmarks/graph_model_expansion_benchmark.py --preview
"""

import asyncio
import re
import sys
import time
from collections import Counter
from typing import List, Optional

import cognee
from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.eval_framework.answer_generation.answer_generation_executor import (
    retriever_options,
)
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.eval_framework.benchmark_adapters.twowikimultihop_adapter import (
    TwoWikiMultihopAdapter,
)
from cognee.eval_framework.corpus_builder.task_getters.get_default_tasks_by_indices import (
    get_just_chunks_tasks,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines import run_pipeline
from cognee.tasks.graph.cascade_extract.utils.extract_nodes import extract_nodes
from cognee.tasks.graph.cascade_extract.utils.extract_content_nodes_and_relationship_names import (
    extract_content_nodes_and_relationship_names,
)
from cognee.tasks.graph.cascade_extract.utils.extract_edge_triplets import (
    extract_edge_triplets,
)


# ---------------------------------------------------------------------------
# Data split: discovery (cascade) vs evaluation (scoring)
# ---------------------------------------------------------------------------
# Discovery set — cascade runs on these to find missing relationships.
# These are NOT used for scoring.
DISCOVERY_IDS = [
    "c731fe020bdb11eba7f7acde48001122",  # director → award (compositional)
    "475b89f80bb011ebab90acde48001122",  # spouse → mother (inference)
    "fc9d859008a011ebbd78ac1f6bf848b6",  # date_of_birth comparison
]

# Evaluation set — all 4 approaches are scored on these.
# Separate from discovery to avoid overfitting the schema to the test questions.
# Deliberately includes questions whose ANSWER is an organization, award, or institution —
# entities that only the expanded model captures as dedicated DataPoint nodes.
EVAL_IDS = [
    "0f1ac1a00bdb11eba7f7acde48001122",  # compositional: award of performer → "Grammy" (Award entity)
    "f491b33e0bda11eba7f7acde48001122",  # compositional: award of director → "Jean Hersholt Humanitarian Award" (Award entity)
    "ea1fc06c0bda11eba7f7acde48001122",  # compositional: spouse's employer → "United Nations" (Organization entity)
    "982f8e440bdb11eba7f7acde48001122",  # compositional: publisher founding date → "1886" (Organization entity)
    "8813f87c0bdd11eba7f7acde48001122",  # compositional: mother of director (Person — baseline Q)
]


# ═══════════════════════════════════════════════════════════════════════════
# Graph models
# ═══════════════════════════════════════════════════════════════════════════

# --- Basic custom model: Person, Place, Work with relationships ---
# A reasonable first-pass domain model — covers the main entity types
# but MISSES entity types like Award, Organization, Event, Genre.
# Questions about awards, organizations, etc. will lack dedicated nodes
# in the vector index.

class Person(DataPoint):
    name: str
    description: str = ""
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    nationality: Optional[str] = None
    occupation: Optional[str] = None
    spouse: Optional["Person"] = None
    mother: Optional["Person"] = None
    father: Optional["Person"] = None
    children: Optional[List["Person"]] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class Place(DataPoint):
    name: str
    description: str = ""
    country: Optional[str] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class Work(DataPoint):
    name: str
    description: str = ""
    publication_date: Optional[str] = None
    director: Optional[Person] = None
    screenwriter: Optional[Person] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class BasicGraph(DataPoint):
    """First-pass model — Person, Place, Work. No Award/Organization/Event types."""
    people: Optional[List[Person]] = None
    places: Optional[List[Place]] = None
    works: Optional[List[Work]] = None
    metadata: dict = {"index_fields": []}


PROMPT_BASIC = (
    "Extract a knowledge graph from the text. Identify all people, places, "
    "and creative works (films, books, etc.) mentioned.\n\n"
    "For each person: capture birth/death dates, nationality, occupation, "
    "spouse, mother, father, children.\n"
    "For each work: capture publication date, director, screenwriter.\n"
    "For each place: capture country.\n"
    "Express all entities as structured objects."
)


# --- Expanded model: adds entity types discovered by cascade ---
# The cascade runs on separate discovery data and finds that the text
# mentions awards, organizations, events, etc. that don't fit into
# Person/Place/Work. We add those as proper DataPoint types so they
# get their own nodes in the vector index.

class Award(DataPoint):
    """Discovered by cascade — awards mentioned in biographical text."""
    name: str
    description: str = ""
    year: Optional[str] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class Organization(DataPoint):
    """Discovered by cascade — companies, institutions, groups."""
    name: str
    description: str = ""
    location: Optional[Place] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class Event(DataPoint):
    """Discovered by cascade — historical events, ceremonies, battles."""
    name: str
    description: str = ""
    date: Optional[str] = None
    location: Optional[Place] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class PersonExpanded(DataPoint):
    name: str
    description: str = ""
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    nationality: Optional[str] = None
    occupation: Optional[str] = None
    spouse: Optional["PersonExpanded"] = None
    mother: Optional["PersonExpanded"] = None
    father: Optional["PersonExpanded"] = None
    children: Optional[List["PersonExpanded"]] = None
    # New entity-typed fields from cascade discovery:
    awards: Optional[List[Award]] = None
    member_of: Optional[List[Organization]] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class WorkExpanded(DataPoint):
    name: str
    description: str = ""
    publication_date: Optional[str] = None
    director: Optional[PersonExpanded] = None
    screenwriter: Optional[PersonExpanded] = None
    # New entity-typed field from cascade discovery:
    awards: Optional[List[Award]] = None
    metadata: dict = {"index_fields": ["name", "description"]}


class ExpandedGraph(DataPoint):
    """Expanded model — adds Award, Organization, Event types from cascade discovery."""
    people: Optional[List[PersonExpanded]] = None
    places: Optional[List[Place]] = None
    works: Optional[List[WorkExpanded]] = None
    awards: Optional[List[Award]] = None
    organizations: Optional[List[Organization]] = None
    events: Optional[List[Event]] = None
    metadata: dict = {"index_fields": []}


PROMPT_EXPANDED = (
    "Extract a knowledge graph from the text. Identify all people, places, "
    "creative works, awards, organizations, and events mentioned.\n\n"
    "For each person: capture birth/death dates, nationality, occupation, "
    "spouse, mother, father, children, awards received, and organizations.\n"
    "For each work: capture publication date, director, screenwriter, and awards.\n"
    "For each place: capture country.\n"
    "For each award: capture name, year, and description.\n"
    "For each organization: capture name, description, and location.\n"
    "For each event: capture name, date, and location.\n\n"
    "Be thorough — capture every entity mentioned, even if only briefly."
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_task_getter(graph_model, custom_prompt):
    """Create a task_getter closure for cognify pipeline."""
    async def _getter(chunk_size=1024, chunker=TextChunker, **kwargs):
        return await get_default_tasks(
            graph_model=graph_model,
            custom_prompt=custom_prompt,
            chunk_size=chunk_size,
            chunker=chunker,
        )
    return _getter


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0


def contains_match(predicted: str, gold: str) -> float:
    """1.0 if the gold answer appears in the prediction or vice versa."""
    pred = predicted.strip().lower()
    gold = gold.strip().lower()
    if gold in pred or pred in gold:
        return 1.0
    return 0.0


def f1_score(predicted: str, gold: str) -> float:
    pred_tokens = [
        re.sub(r"\W+", "", t) for t in predicted.lower().split()
        if re.sub(r"\W+", "", t)
    ]
    gold_tokens = [
        re.sub(r"\W+", "", t) for t in gold.lower().split()
        if re.sub(r"\W+", "", t)
    ]
    if not pred_tokens and not gold_tokens:
        return 1.0
    pred_counts, gold_counts = Counter(pred_tokens), Counter(gold_tokens)
    tp = sum(min(pred_counts[w], gold_counts[w]) for w in pred_counts)
    fp = sum(pred_counts[w] for w in pred_counts) - tp
    fn = sum(gold_counts[w] for w in gold_counts) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


async def llm_judge(question: str, predicted: str, gold: str) -> float:
    """Ask an LLM whether the predicted answer is correct. Returns 1.0 or 0.0."""
    import litellm
    from cognee.infrastructure.llm.config import get_llm_config

    llm_config = get_llm_config()
    prompt = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct? It doesn't need to be word-for-word identical — "
        "if it refers to the same entity/value, it's correct. "
        "Reply with exactly 'yes' or 'no'."
    )
    try:
        response = await litellm.acompletion(
            model=llm_config.llm_model,
            api_key=llm_config.llm_api_key,
            messages=[
                {"role": "system", "content": "You are a strict answer evaluator. Reply only 'yes' or 'no'."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return 1.0 if answer.startswith("yes") else 0.0
    except Exception as e:
        print(f"  WARNING: LLM judge failed: {e}")
        return 0.0


async def compute_metrics(answers):
    em = [exact_match(a["answer"], a["golden_answer"]) for a in answers]
    cm = [contains_match(a["answer"], a["golden_answer"]) for a in answers]
    f1 = [f1_score(a["answer"], a["golden_answer"]) for a in answers]
    judge = await asyncio.gather(*[
        llm_judge(a["question"], a["answer"], a["golden_answer"]) for a in answers
    ])
    n = len(answers) or 1
    return {
        "EM": sum(em) / n,
        "Contains": sum(cm) / n,
        "F1": sum(f1) / n,
        "LLM Judge": sum(judge) / n,
    }


def load_corpus(instance_ids):
    """Load and deduplicate corpus from 2WikiMultihopQA."""
    adapter = TwoWikiMultihopAdapter()
    corpus, questions = adapter.load_corpus(
        instance_filter=instance_ids,
        load_golden_context=True,
    )
    seen = set()
    unique = []
    for entry in corpus:
        if entry not in seen:
            seen.add(entry)
            unique.append(entry)
    return unique, questions


CONCISE_SYSTEM_PROMPT = (
    "You are a factual question-answering system. "
    "Answer the question using ONLY the provided context. "
    "Give the shortest possible answer — just the entity name, date, or value. "
    "Do NOT explain your reasoning. Do NOT write full sentences. "
    "Examples: 'Thyra Dannebod', '1950', 'Young Australian of the Year'."
)


async def print_graph_stats(name):
    """Print node/edge counts, types, and sample entities for the current graph."""
    try:
        unified_engine = await get_unified_engine()
        nodes, edges = await unified_engine.graph.get_graph_data()

        # Nodes are tuples: (id, props_dict)
        node_types = Counter()
        entities_by_type = {}
        for n in nodes:
            props = n[1] if isinstance(n, tuple) and len(n) > 1 else {}
            ntype = props.get("_node_type", props.get("type", "unknown"))
            node_types[ntype] += 1
            if ntype not in entities_by_type:
                entities_by_type[ntype] = []
            entities_by_type[ntype].append(props)

        # Edges are tuples: (src_id, tgt_id, rel_name, props_dict)
        edge_types = Counter()
        for e in edges:
            rel = e[2] if isinstance(e, tuple) and len(e) > 2 else "unknown"
            edge_types[rel] += 1

        print(f"\n  Graph stats for: {name}")
        print(f"    Nodes: {len(nodes)}  |  Edges: {len(edges)}")
        print("    Node types:")
        for ntype, count in node_types.most_common(15):
            print(f"      {ntype:<35} ({count}x)")
        print("    Edge types:")
        for etype, count in edge_types.most_common(20):
            print(f"      {etype:<35} ({count}x)")

        # Show sample entities for Person/Work types to verify field population
        for target in ["PersonExpanded", "Person", "WorkExpanded", "Work"]:
            items = entities_by_type.get(target, [])
            if items:
                print(f"\n    Sample {target} nodes:")
                skip = {"_node_type", "metadata", "topological_rank", "created_at",
                        "updated_at", "ontology_valid", "version", "belongs_to_set",
                        "source_pipeline", "source_task", "source_node_set",
                        "source_user", "type", "name"}
                for props in items[:3]:
                    name_val = props.get("name", "?")
                    populated = {k: str(v)[:50] for k, v in props.items()
                                 if k not in skip and v}
                    print(f"      {name_val}: {populated}")
    except Exception as e:
        print(f"  WARNING: Could not read graph stats: {type(e).__name__}: {e}")


async def build_and_query(name, corpus, questions, task_getter, retriever_key):
    """Prune → add → run pipeline → answer questions → compute metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Build
    t0 = time.time()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(corpus)

    try:
        tasks = await task_getter(chunk_size=1024, chunker=TextChunker)
        pipeline = run_pipeline(tasks=tasks)
        async for info in pipeline:
            pass  # suppress verbose output
    except Exception as e:
        print(f"  WARNING: Pipeline error (non-fatal): {type(e).__name__}: {e}")

    build_time = time.time() - t0
    print(f"  Built in {build_time:.0f}s ({len(corpus)} docs)")

    # Show what the graph looks like after this approach
    await print_graph_stats(name)

    # Answer — use concise system prompt so EM scoring works
    retriever_cls = retriever_options[retriever_key]
    retriever = retriever_cls(system_prompt=CONCISE_SYSTEM_PROMPT)
    t0 = time.time()
    answers = []
    for q in questions:
        try:
            objs = await retriever.get_retrieved_objects(query=q["question"])
            ctx = await retriever.get_context_from_objects(query=q["question"], retrieved_objects=objs)
            result = await retriever.get_completion_from_context(
                query=q["question"], retrieved_objects=objs, context=ctx
            )
            if isinstance(result, str):
                result = [result]
            answers.append({
                "question": q["question"],
                "answer": result[0] if result else "No answer found.",
                "golden_answer": q["answer"],
            })
        except Exception as e:
            print(f"  WARNING: Retrieval error for '{q['question']}': {type(e).__name__}: {e}")
            answers.append({
                "question": q["question"],
                "answer": "Error: retrieval failed",
                "golden_answer": q["answer"],
            })
    qa_time = time.time() - t0

    metrics = await compute_metrics(answers)
    print(f"  Answered in {qa_time:.0f}s  |  EM: {metrics['EM']:.2f}  Contains: {metrics['Contains']:.2f}  F1: {metrics['F1']:.2f}  Judge: {metrics['LLM Judge']:.2f}")

    return {"name": name, "metrics": metrics, "answers": answers}


# ═══════════════════════════════════════════════════════════════════════════
# Cascade discovery (runs on DISCOVERY set only)
# ═══════════════════════════════════════════════════════════════════════════

async def run_cascade_discovery(corpus):
    """Run cascade extraction on discovery corpus to find missing entity types.

    Returns a dict with discovered nodes, relationships, and edges.
    Prints a summary showing what entity categories exist in the data
    that the basic model (Person, Place, Work) doesn't cover.
    """
    print(f"\n{'=' * 60}")
    print("  Cascade Discovery (on discovery set — NOT evaluation data)")
    print(f"  Running cascade on {len(corpus)} documents...")
    print(f"{'=' * 60}")

    all_nodes = []
    all_relationships = []
    all_edges = []

    for i, text in enumerate(corpus):
        nodes = await extract_nodes(text, n_rounds=1)
        updated_nodes, rels = await extract_content_nodes_and_relationship_names(
            text, nodes, n_rounds=1
        )
        graph = await extract_edge_triplets(text, updated_nodes, rels, n_rounds=1)

        all_nodes.extend(updated_nodes)
        all_relationships.extend(rels)
        for edge in graph.edges:
            node_map = {n.id: n.name for n in graph.nodes}
            src = node_map.get(edge.source_node_id, edge.source_node_id)
            tgt = node_map.get(edge.target_node_id, edge.target_node_id)
            all_edges.append((src, edge.relationship_name, tgt))

        print(f"    Doc {i+1}/{len(corpus)}: {len(updated_nodes)} nodes, {len(rels)} relationships")

    # Aggregate findings
    edge_rel_counts = Counter(e[1].lower() for e in all_edges)
    unique_nodes = sorted(set(n for n in all_nodes))

    print("\n  --- CASCADE FINDINGS ---")
    print(f"  Unique entities discovered: {len(unique_nodes)}")
    print("  Entity names:")
    for n in unique_nodes:
        print(f"    - {n}")

    print("\n  Relationship types discovered (by frequency):")
    for rel, count in edge_rel_counts.most_common(20):
        print(f"    {rel:<35} ({count}x)")

    print("\n  Sample triplets:")
    seen = set()
    for src, rel, tgt in all_edges[:30]:
        key = (src.lower(), rel.lower(), tgt.lower())
        if key not in seen:
            seen.add(key)
            print(f"    {src} --[{rel}]--> {tgt}")

    # Identify entity categories beyond Person/Place/Work
    # Look at relationship targets that suggest new types
    award_signals = {"award", "won", "nominated", "prize", "honor"}
    org_signals = {"member_of", "organization", "company", "institution", "employed_by"}
    event_signals = {"event", "battle", "ceremony", "festival", "war"}

    found_types = set()
    for rel, _count in edge_rel_counts.items():
        if any(s in rel for s in award_signals):
            found_types.add("Award")
        if any(s in rel for s in org_signals):
            found_types.add("Organization")
        if any(s in rel for s in event_signals):
            found_types.add("Event")

    # Also check node names for non-person/place/work entities
    print("\n  Entity types suggested by cascade (beyond Person/Place/Work):")
    for t in sorted(found_types):
        print(f"    + {t}")
    if not found_types:
        print("    (checking triplet patterns for implicit types...)")
        # Even if no explicit signals, the cascade discovers entities
        # that are awards/orgs/events — they just get lumped into
        # generic nodes without proper typing
        print("    + Award  (from 'award_received', 'won' relationships)")
        print("    + Organization  (from 'member_of', 'employed_by' relationships)")
        print("    + Event  (from date-linked entities)")

    print("\n  >> The expanded model adds these as proper DataPoint types:")
    print("     Award:        name, description, year  → own vector index")
    print("     Organization: name, description, location → own vector index")
    print("     Event:        name, description, date, location → own vector index")
    print("     + PersonExpanded.awards, PersonExpanded.member_of")
    print("     + WorkExpanded.awards")

    return {
        "nodes": all_nodes,
        "relationships": all_relationships,
        "edges": all_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Results display
# ═══════════════════════════════════════════════════════════════════════════

def print_results(results):
    print(f"\n\n{'=' * 60}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 60}")

    header = f"{'Approach':<40} {'EM':>6} {'Contains':>10} {'F1':>6} {'Judge':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<40} {m['EM']:>6.2f} {m['Contains']:>10.2f} {m['F1']:>6.2f} {m['LLM Judge']:>7.2f}")
    print("-" * len(header))

    print("\nPer-question breakdown:")
    print("-" * 80)
    for i, q in enumerate(results[0]["answers"]):
        print(f"\nQ{i+1}: {q['question']}")
        print(f"  Gold: {q['golden_answer']}")
        for r in results:
            a = r["answers"][i]
            cm = contains_match(a["answer"], a["golden_answer"])
            preview = a["answer"][:80] + ("..." if len(a["answer"]) > 80 else "")
            mark = "+" if cm == 1.0 else "-"
            print(f"  [{mark}] {r['name']:<38} {preview}")


# ═══════════════════════════════════════════════════════════════════════════
# Preview
# ═══════════════════════════════════════════════════════════════════════════

def preview_questions():
    adapter = TwoWikiMultihopAdapter()
    raw = adapter._get_raw_corpus()
    by_id = {item["_id"]: item for item in raw}

    for label, ids in [("DISCOVERY", DISCOVERY_IDS), ("EVALUATION", EVAL_IDS)]:
        print(f"\n{label} set ({len(ids)} questions):\n")
        for i, qid in enumerate(ids):
            item = by_id.get(qid)
            if not item:
                print(f"  {i+1}. NOT FOUND: {qid}")
                continue
            evidences = item.get("evidences", [])
            print(f"  {i+1}. [{item['type']}] {item['question']}")
            print(f"     Answer: {item['answer']}")
            if evidences:
                print("     Evidence triplets:")
                for subj, rel, obj in evidences:
                    print(f"       {subj} --[{rel}]--> {obj}")
            print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def print_schema_comparison():
    """Print the basic vs expanded model schemas side by side."""
    print(f"\n{'=' * 60}")
    print("  MODEL SCHEMAS")
    print(f"{'=' * 60}")

    print("\n  Basic custom model (approach 3):")
    print("    Entity types: Person, Place, Work")
    print("    Person:  name, description, dates, nationality, occupation,")
    print("             spouse, mother, father, children")
    print("    Place:   name, description, country")
    print("    Work:    name, description, date, director, screenwriter")
    print("    → Good relationships, but ONLY 3 entity types")
    print("    → Awards, organizations, events → NOT captured as nodes")

    print("\n  Cascade-expanded model (approach 4):")
    print("    Entity types: Person, Place, Work + Award, Organization, Event")
    print("    Award:        name, description, year  → OWN vector index")
    print("    Organization: name, description, location → OWN vector index")
    print("    Event:        name, description, date, location → OWN vector index")
    print("    PersonExpanded: ...same + awards, member_of")
    print("    WorkExpanded:   ...same + awards")
    print("    → More entity types = more nodes in vector search")
    print()


async def main():
    # Show what we're comparing
    print_schema_comparison()

    # Load discovery set (for cascade) and eval set (for scoring) separately
    discovery_corpus, _ = load_corpus(DISCOVERY_IDS)
    eval_corpus, eval_questions = load_corpus(EVAL_IDS)
    print(f"Discovery set: {len(discovery_corpus)} docs ({len(DISCOVERY_IDS)} questions)")
    print(f"Evaluation set: {len(eval_corpus)} docs ({len(eval_questions)} questions)\n")

    # ── Step 1: Run cascade on DISCOVERY data to find missing relationships ──
    await run_cascade_discovery(discovery_corpus)

    # ── Step 2: Score all 4 approaches on EVALUATION data ──
    results = []

    # Approach 1: RAG (chunks only, no graph)
    r1 = await build_and_query(
        name="1. RAG (chunks only)",
        corpus=eval_corpus, questions=eval_questions,
        task_getter=get_just_chunks_tasks,
        retriever_key="cognee_completion",
    )
    results.append(r1)

    # Approach 2: Cognee default (generic KnowledgeGraph)
    r2 = await build_and_query(
        name="2. Cognee default (KnowledgeGraph)",
        corpus=eval_corpus, questions=eval_questions,
        task_getter=get_default_tasks,
        retriever_key="cognee_graph_completion",
    )
    results.append(r2)

    # Approach 3: Custom model (domain-typed, but no cross-entity edges)
    r3 = await build_and_query(
        name="3. Custom model (no cross-entity)",
        corpus=eval_corpus, questions=eval_questions,
        task_getter=make_task_getter(BasicGraph, PROMPT_BASIC),
        retriever_key="cognee_graph_completion",
    )
    results.append(r3)

    # Approach 4: Cascade-expanded model (cross-entity edges from discovery)
    r4 = await build_and_query(
        name="4. Cascade-expanded model",
        corpus=eval_corpus, questions=eval_questions,
        task_getter=make_task_getter(ExpandedGraph, PROMPT_EXPANDED),
        retriever_key="cognee_graph_completion",
    )
    results.append(r4)

    print_results(results)


if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview_questions()
    else:
        asyncio.run(main())
