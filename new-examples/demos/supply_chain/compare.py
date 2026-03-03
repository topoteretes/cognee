"""Side-by-side comparison: Baseline (tool-calling) agent vs Cognee agent.

Runs independent questions through both agents. Each question is a cold
start (no conversation history from prior steps). Each question requires
combining data from multiple source systems to answer correctly.

Questions are deliberately phrased in business language (not table/column
names) and often require negative reasoning, temporal ordering, or
policy-vs-exception reconciliation. Several questions use vocabulary that
does NOT match CSV column values (e.g. "ocean freight" vs "Sea FCL",
"overdue" vs "DELAYED") to test semantic understanding.

The data includes ~250 shipments, ~200 customer orders, ~240 POs, and
~530 tracking events — enough to overwhelm in-context table scanning.

Usage:
    .venv/bin/python new-examples/demos/supply_chain/compare.py
    .venv/bin/python new-examples/demos/supply_chain/compare.py --step 3
    .venv/bin/python new-examples/demos/supply_chain/compare.py --skip-build --step 7
    .venv/bin/python new-examples/demos/supply_chain/compare.py --max-rounds 15 --summarize
"""

import asyncio
import io
import os
import sys
import textwrap
import time
from datetime import datetime

import cognee
from cognee.modules.users.methods import get_default_user
from cognee.memify_pipelines.create_triplet_embeddings import create_triplet_embeddings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_agent
import cognee_agent
from toy_data import (
    get_disruption_alert,
    get_planner_notes_data,
    get_policy_document_paths,
)
from csv_to_graph import build_graph_from_csvs
from models import SupplyChainContext

# ── Monkey-patch: prevent empty strings reaching the embedding API ────
# Cognee's index_data_points only skips None, not "". Internal DataPoint
# types (EdgeType, copied models) bypass our SupplyChainDataPoint validator,
# so we intercept here and coerce any empty index field to " ".
from importlib import import_module as _im

_idx_mod = _im("cognee.tasks.storage.index_data_points")
_adp_mod = _im("cognee.tasks.storage.add_data_points")
_ige_mod = _im("cognee.tasks.storage.index_graph_edges")

_original_index_data_points = _idx_mod.index_data_points


async def _safe_index_data_points(data_points):
    for dp in data_points:
        for f in dp.metadata.get("index_fields", []):
            val = getattr(dp, f, None)
            if isinstance(val, str) and not val.strip():
                object.__setattr__(dp, f, "n/a")
    return await _original_index_data_points(data_points)


_idx_mod.index_data_points = _safe_index_data_points
_adp_mod.index_data_points = _safe_index_data_points
_ige_mod.index_data_points = _safe_index_data_points
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "comparison_output.log")

_log_file = None


def _echo(msg: str = "") -> None:
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()


def _print_section(title: str) -> None:
    sep = "=" * 70
    _echo("")
    _echo(sep)
    _echo(f"  {title}")
    _echo(sep)


def _wrap(text: str, width: int = 90) -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped)


# ── Questions ─────────────────────────────────────────────────────────
#
# Each question requires combining data from MULTIPLE CSVs.
# The "sources_needed" comment explains what the agent must connect.

SCENARIO = [
    # ── Original questions (rephrased where noted) ────────────────────
    {
        "id": 1,
        "title": "Shipment delay → customer impact → financial penalty",
        "question": (
            "Our ocean freight consignment SHP-006 is overdue. Which purchase "
            "order does it belong to, which customer order depends on it, what "
            "is the daily penalty, and who is the customer?"
        ),
        "why_hard": (
            "REPHRASED: 'ocean freight' (CSV: 'Sea FCL'), 'overdue' (CSV: 'DELAYED'). "
            "Baseline filters may return 0 rows. Also requires chaining SHP-006 → "
            "PO-2024-006 → CUST-ORD-1042 → Nexus Robotics $244/day across 3 sources."
        ),
        "expected": {
            "must_contain": ["SHP-006", "PO-2024-006", "CUST-ORD-1042", "Nexus Robotics", "244"],
            "must_not_contain": [],
        },
    },
    {
        "id": 2,
        "title": "Sole-source risk → BOM cascade → downstream impact",
        "question": (
            "SKU-002 is sole-sourced from one supplier. "
            "Which supplier is it, what is their reliability score, "
            "and which other SKUs depend on SKU-002 through the Bill of Materials?"
        ),
        "why_hard": (
            "Requires matching SKU-002's primary_supplier (SUPP-002) to suppliers.csv "
            "for the name and reliability score, plus checking ALL SKU BOMs to find "
            "which SKUs list SKU-002 as a component (SKU-001 uses 1×SKU-002). "
            "The baseline must cross-reference two CSVs and parse BOM strings."
        ),
        "expected": {
            "must_contain": ["SUPP-002", "Rhine Components GmbH", "3.2", "SKU-001"],
            "must_not_contain": [],
        },
    },
    {
        "id": 3,
        "title": "Supplier failure — business language, no IDs given",
        "question": (
            "If our primary German capacitor supplier went bankrupt today, "
            "which purchase orders, shipments, and customer orders would be "
            "affected? List specific IDs and the total financial exposure."
        ),
        "why_hard": (
            "REPHRASED: No supplier ID or name given. 'German' (CSV: 'Germany') "
            "is a substring mismatch if filtered literally. 'capacitor supplier' "
            "requires knowing SUPP-002 supplies SKU-002 (Aluminum Housing) — the "
            "baseline must first identify the supplier, then trace the chain. "
            "With 4 Rhine entities, it may pick the wrong one."
        ),
        "expected": {
            "must_contain": ["SUPP-002", "Rhine Components GmbH", "PO-2024", "CUST-ORD", "SKU-002"],
            "must_not_contain": [],
        },
    },
    {
        "id": 4,
        "title": "Expedite decision → policy compliance → cost justification",
        "question": (
            "A planner expedited SHP-003 via air freight. "
            "What was the rationale, what was the cost, "
            "and does our expedite policy allow air freight for this case? "
            "What customer order was at stake?"
        ),
        "why_hard": (
            "Must find the expedite decision in planner_decisions, match SHP-003 "
            "in shipments, find which customer order lists SHP-003 in "
            "fulfillment_shipments (CUST-ORD-1042), and check the expedite policy "
            "document. Four different data sources including a text document."
        ),
        "expected": {
            "must_contain": ["SHP-003", "CUST-ORD-1042", "Nexus Robotics"],
            "must_not_contain": [],
        },
    },
    {
        "id": 5,
        "title": "Buffer inventory check — vocabulary mismatch",
        "question": (
            "Customer CUST-ORD-1042 needs 200 units of SKU-001 and 400 units "
            "of SKU-003. What is the current buffer inventory of each SKU "
            "across all facilities, and is there enough on hand to cover "
            "this order if the in-transit consignments are held up?"
        ),
        "why_hard": (
            "REPHRASED: 'buffer inventory' (CSV: 'safety_stock_site_*'), "
            "'facilities' (CSV: site names), 'in-transit consignments' (CSV: "
            "'shipments' with status 'IN TRANSIT'). Baseline must map business "
            "vocabulary to CSV columns. SKU-001: 100+150+200=450 (ok for 200). "
            "SKU-003: 200+250+300=750 (ok for 400)."
        ),
        "expected": {
            "must_contain": ["SKU-001", "SKU-003", "450", "750"],
            "must_not_contain": [],
        },
    },
    {
        "id": 6,
        "title": "Contradictory data across sources — root cause override",
        "question": (
            "What was the real root cause of SHP-002's delay — "
            "was it customs clearance or something else?"
        ),
        "why_hard": (
            "shipments.csv says 'customs clearance'. But planner_decisions DEC-002 "
            "and feedback_corrections correct this to 'Antwerp berth congestion'. "
            "The baseline must call lookup_feedback_corrections (separate tool) "
            "and recognize the correction overrides the TMS record. If it only "
            "calls lookup_shipments, it gives the wrong answer."
        ),
        "expected": {
            "must_contain": ["berth congestion", "Antwerp"],
            "must_not_contain": [],
        },
    },
    {
        "id": 7,
        "title": "DG carrier compliance — vocabulary mismatch",
        "question": (
            "We need to move SKU-003 from our European port to our West Coast "
            "facility urgently. SKU-003 requires hazmat-rated carriers. "
            "Which ocean carriers serve that route, which are certified for "
            "this cargo class, and what does our policy say about carrier "
            "selection for regulated materials?"
        ),
        "why_hard": (
            "REPHRASED: 'European port' (CSV: 'Hamburg'), 'West Coast facility' "
            "(CSV: 'LA'), 'ocean carriers' (CSV: carrier_performance with lane), "
            "'hazmat-rated' (CSV: 'dg_approved'), 'regulated materials' (policy: "
            "'dangerous_goods'). Every filter term mismatches. Baseline must map "
            "'European port'→Hamburg, 'West Coast'→LA, 'hazmat'→DG, "
            "'regulated materials'→'dangerous_goods'. If any mapping fails, "
            "it gets 0 results. Cognee's semantic search handles synonyms."
        ),
        "expected": {
            "must_contain": ["SKU-003", "UN3480", "Hamburg", "LA"],
            "must_not_contain": [],
        },
    },
    {
        "id": 8,
        "title": "Sole-source mitigation — policy + supplier + SKU + customer",
        "question": (
            "SKU-005 is sole-sourced. What is our company policy for mitigating "
            "sole-source risk? Which customer orders need SKU-005, and what "
            "is the current safety stock situation?"
        ),
        "why_hard": (
            "Must identify SKU-005 as sole-sourced from SUPP-005 (skus.csv), "
            "find customer orders requiring SKU-005, check safety stock "
            "(east 80, west 90, dc 100 = 270 total), and retrieve the "
            "sole_source_mitigation policy. Four sources."
        ),
        "expected": {
            "must_contain": ["SKU-005", "SUPP-005", "sole-source", "270"],
            "must_not_contain": [],
        },
    },
    {
        "id": 9,
        "title": "Temporal timeline — reconstruct the full chronology of a shipment",
        "question": (
            "Give me the complete timeline for SHP-002: every tracking event "
            "in chronological order, plus any planner decisions, feedback "
            "corrections, and disruption events that relate to it. "
            "For each entry, include the date and what happened."
        ),
        "why_hard": (
            "The baseline must call at least 4 tools (tracking events, shipments, "
            "planner notes, feedback corrections, and disruption events) and merge "
            "everything into a single chronological timeline. If it misses any "
            "source, the timeline is incomplete. With the new parameterized tools, "
            "it must also pass the right filters. Cognee's graph links all these "
            "entities to SHP-002 through typed edges."
        ),
        "expected": {
            "must_contain": ["SHP-002", "FB-001", "berth congestion"],
            "must_not_contain": [],
        },
    },
    # ── Harder questions (business language, negative reasoning, etc.) ─
    {
        "id": 10,
        "title": "Indirect reference — no IDs, business language only",
        "question": (
            "The shipment we air-expedited for Nexus — which purchase order "
            "did it originally belong to, what was the total extra cost, "
            "and was the expedite compliant with our freight escalation policy?"
        ),
        "why_hard": (
            "Zero IDs in the question. The baseline must figure out that "
            "'air-expedited for Nexus' maps to DEC-001/SHP-003 by searching "
            "planner decisions AND customer orders for 'Nexus', then chain to "
            "the PO and policy. Cognee's graph links the customer name, "
            "shipment, decision, and policy in one connected subgraph."
        ),
        "expected": {
            "must_contain": ["SHP-003", "Nexus Robotics", "CUST-ORD-1042"],
            "must_not_contain": [],
        },
    },
    {
        "id": 11,
        "title": "Negative reasoning — sole-sourced SKUs without a backup PO",
        "question": (
            "Which of our sole-sourced SKUs currently have no backup or "
            "contingency purchase order? For each one, tell me the supplier, "
            "the customer orders at risk, and whether our sole-source "
            "mitigation policy is being followed."
        ),
        "why_hard": (
            "Requires negative reasoning: the agent must find all sole-sourced "
            "SKUs, then check purchase_orders AND planner_decisions for each to "
            "confirm no backup exists. The key insight is proving something "
            "does NOT exist. Cognee surfaces the gap because a missing edge "
            "is visible in a graph."
        ),
        "expected": {
            "must_contain": ["sole-source", "SKU-002", "SKU-005"],
            "must_not_contain": [],
        },
    },
    {
        "id": 12,
        "title": "Policy exception audit — did we break our own rules?",
        "question": (
            "Our expedite policy says air freight is only justified when the "
            "daily penalty exposure exceeds the freight premium. Did we ever "
            "expedite a shipment where this condition was NOT met? If so, "
            "which shipment, what was the penalty exposure, what was the "
            "freight cost, and who approved it?"
        ),
        "why_hard": (
            "The agent must read the expedite policy, find all expedite "
            "decisions in planner_decisions, look up the penalty exposure "
            "from customer_orders for each linked shipment, and compare: "
            "was freight cost > penalty savings? This is a compliance audit "
            "across 4 sources that requires arithmetic and rule interpretation."
        ),
        "expected": {
            "must_contain": ["SHP-003", "4,500", "244", "penalty"],
            "must_not_contain": [],
        },
    },
    # ── Disambiguation: misleading shipment + supplier name collision ──
    {
        "id": 13,
        "title": "Misleading premise + multi-entity disambiguation on same lane",
        "question": (
            "Our monitoring dashboard flagged SHP-016 as the shipment "
            "fulfilling our aerospace customer's lithium cell order on the "
            "Hamburg→LA lane. Based on this, the team calculated a penalty "
            "exposure of $660/day. However, a previous agent feedback "
            "correction noted that agents have confused shipments on this "
            "lane before. Is SHP-016 really the aerospace customer's "
            "shipment? If not, identify the correct shipment, the correct "
            "customer order, the actual daily penalty, the purchase order, "
            "and which Rhine Components entity is the supplier (we have two "
            "with similar names). Also, has a planner documented this "
            "disambiguation risk?"
        ),
        "why_hard": (
            "SHP-016 is WRONG — it's for the defense contractor (CUST-ORD-1098, "
            "$660/day). The AEROSPACE customer is CUST-ORD-1115 (Aerospace "
            "Components Inc), fulfilled by SHP-021 ($960/day). FB-009 documents "
            "this exact confusion. The baseline must call 5+ tools and resist "
            "trusting the premise."
        ),
        "expected": {
            "must_contain": ["SHP-021", "CUST-ORD-1115", "Aerospace Components", "960", "FB-009"],
            "must_not_contain": [],
        },
    },
    # ── Multiple entities on same line ────────────────────────────────
    {
        "id": 14,
        "title": "Same-vessel table — four MSC Lucia shipments, one row each",
        "question": (
            "We have four shipments on the same vessel (MSC Lucia) on the "
            "Hamburg→LA lane, all carrying lithium cells (SKU-003). For each "
            "of the four, give: (1) shipment ID, (2) customer name, "
            "(3) customer order ID, (4) daily penalty in $/day, and "
            "(5) purchase order number. Present them in a table with one "
            "row per shipment so we can see the differences."
        ),
        "why_hard": (
            "SHP-016, SHP-017, SHP-021, SHP-022 share vessel, lane, carrier, SKU. "
            "Baseline must join shipments → customer_orders → purchase_orders and "
            "assign each row correctly. Cognee's graph edges per shipment give "
            "unambiguous disambiguation."
        ),
        "expected": {
            "must_contain": ["SHP-016", "SHP-017", "SHP-021", "SHP-022"],
            "must_not_contain": [],
        },
    },
    {
        "id": 15,
        "title": "Same lane — which shipment has highest penalty?",
        "question": (
            "Among the Hamburg→LA Maersk lithium cell (SKU-003) shipments, "
            "which one has the highest daily penalty exposure? Give that "
            "shipment's ID, the customer name, order ID, daily penalty, and "
            "purchase order number."
        ),
        "why_hard": (
            "Correct: SHP-021, Aerospace Components Inc, CUST-ORD-1115, "
            "$960/day, PO-2024-019. Baseline must join shipments to "
            "customer_orders to get penalties."
        ),
        "expected": {
            "must_contain": [
                "SHP-021",
                "Aerospace Components",
                "CUST-ORD-1115",
                "960",
                "PO-2024-019",
            ],
            "must_not_contain": [],
        },
    },
    {
        "id": 16,
        "title": "Same-name suppliers — table of four (Rhine + Seoul Micro)",
        "question": (
            "We have two suppliers with 'Rhine' in the name and two with "
            "'Seoul Micro' in the name. For each of the four, give: supplier "
            "ID, full legal name, country, reliability score, and one "
            "purchase order that uses them. Present in a table with one row "
            "per supplier."
        ),
        "why_hard": (
            "SUPP-002 (Rhine GmbH), SUPP-009 (Rhine Ltd), SUPP-004 (Seoul "
            "Micro Ltd), SUPP-010 (Seoul Micro Korea Inc). Baseline must "
            "match POs by supplier_id and attach correct name/country/score."
        ),
        "expected": {
            "must_contain": ["SUPP-002", "SUPP-009", "SUPP-004", "SUPP-010"],
            "must_not_contain": [],
        },
    },
    {
        "id": 17,
        "title": "Same lane — total daily penalty across original SKU-003 shipments",
        "question": (
            "What is the total daily penalty exposure (in $/day) across all "
            "Hamburg→LA Maersk shipments that carry SKU-003? List each "
            "shipment, its customer, and its daily penalty, then give the total."
        ),
        "why_hard": (
            "Many Hamburg→LA SKU-003 shipments now exist. Baseline must list "
            "all and assign correct penalty to each from customer_orders; "
            "missing one or swapping two gives wrong total."
        ),
        "expected": {
            "must_contain": ["SHP-006", "SHP-016", "SHP-021"],
            "must_not_contain": [],
        },
    },
    # ── Scale-dependent questions ─────────────────────────────────────
    {
        "id": 18,
        "title": "Identify the correct shipment for a specific customer (many similar rows)",
        "question": (
            "Which specific shipment fulfills the Aerospace Components Inc "
            "lithium cell order on the Hamburg→LA lane? Give the shipment ID, "
            "purchase order number, daily penalty, and delay status. "
            "Be careful — there are many shipments on that same lane "
            "with the same SKU and carrier."
        ),
        "why_hard": (
            "Many Hamburg→LA SKU-003 shipments look identical. The only way "
            "to find the correct one is to cross-reference customer_orders "
            "with the customer name 'Aerospace Components Inc'. "
            "Correct: SHP-021 / CUST-ORD-1115 / $960/day."
        ),
        "expected": {
            "must_contain": ["SHP-021", "CUST-ORD-1115", "960"],
            "must_not_contain": [],
        },
    },
    {
        "id": 19,
        "title": "Total daily penalty across all Hamburg→LA SKU-003 shipments",
        "question": (
            "What is the total daily penalty exposure (in $/day) across ALL "
            "Hamburg→LA shipments carrying SKU-003? List each shipment "
            "with its customer name and daily penalty, then give the grand "
            "total."
        ),
        "why_hard": (
            "With expanded data there are now 100+ Hamburg→LA SKU-003 shipments. "
            "The baseline must filter shipments, then for EACH one find the "
            "matching customer order from 200 orders by matching "
            "fulfillment_shipments, extract penalty_per_day_usd, and sum. "
            "Missing even one row gives a wrong total."
        ),
        "expected": {
            "must_contain": ["SHP-006", "SHP-021"],
            "must_not_contain": [],
        },
    },
    {
        "id": 20,
        "title": "Disambiguate 4 Rhine-named suppliers",
        "question": (
            "We have multiple suppliers with 'Rhine Components' in the name. "
            "For each one, give: supplier ID, full legal name, country, "
            "reliability score, and one purchase order that references them. "
            "Which Rhine entity has the worst reliability and why?"
        ),
        "why_hard": (
            "4 Rhine suppliers: SUPP-002 (GmbH, Germany, 3.2), SUPP-009 "
            "(Ltd, UK, 4.6), SUPP-013 (Asia, Singapore, 3.8), SUPP-014 "
            "(NA, US, 3.5). Worst reliability: SUPP-002 at 3.2."
        ),
        "expected": {
            "must_contain": ["SUPP-002", "SUPP-009", "SUPP-013", "SUPP-014", "3.2"],
            "must_not_contain": [],
        },
    },
    {
        "id": 21,
        "title": "BOM cascade: all products affected if SKU-003 is unavailable",
        "question": (
            "If SKU-003 (Lithium Battery Cell LBC-4800) becomes completely "
            "unavailable, trace the FULL Bill of Materials cascade: which "
            "assemblies and products are affected at every level? For each "
            "affected end product, list the customer orders that need it "
            "and their daily penalty. What is the total penalty exposure?"
        ),
        "why_hard": (
            "SKU-003 is used by SKU-001, SKU-009, SKU-011, SKU-013, SKU-014, "
            "SKU-016. Those cascade to SKU-010, SKU-012, SKU-015, SKU-017. "
            "4-level cascade. Baseline must parse bom_components strings "
            "recursively across 17 SKUs."
        ),
        "expected": {
            "must_contain": ["SKU-001", "SKU-009", "SKU-011", "SKU-013", "SKU-014", "SKU-016"],
            "must_not_contain": [],
        },
    },
    {
        "id": 22,
        "title": "Root cause override — actual vs TMS delay reason for many shipments",
        "question": (
            "For each of the delayed or at-risk Hamburg→LA shipments, what is "
            "the ACTUAL root cause of delay (not the TMS delay code)? Our "
            "feedback corrections show that TMS codes are often wrong. List "
            "each shipment with both the TMS reason and the corrected reason, "
            "plus the affected customer and daily penalty."
        ),
        "why_hard": (
            "The baseline must: (1) get delayed shipments, (2) call "
            "lookup_feedback_corrections (separate tool) for the real root "
            "causes, (3) get customer orders for penalties. Three separate "
            "tools with large outputs and precise cross-referencing."
        ),
        "expected": {
            "must_contain": ["SHP-002", "berth congestion"],
            "must_not_contain": [],
        },
    },
    # ── NEW: Temporal reasoning questions ─────────────────────────────
    {
        "id": 23,
        "title": "Temporal — were customer orders already late when delivered?",
        "question": (
            "Which customer orders had already passed their delivery deadline "
            "by the time their fulfillment shipment was actually delivered? "
            "For each, give the order ID, customer name, deadline date, "
            "actual delivery date, and how many days late."
        ),
        "why_hard": (
            "Baseline must: (1) get customer_orders (delivery_deadline), "
            "(2) get shipments (actual_delivery, filtering for DELIVERED), "
            "(3) join by fulfillment_shipments, (4) compare dates and compute "
            "days late. Cross-payload date arithmetic with 200+ customer "
            "orders and 250 shipments. Cognee's graph has CustomerOrder→Shipment "
            "edges with both dates in context."
        ),
        "expected": {
            "must_contain": ["CUST-ORD-1042", "Nexus", "days"],
            "must_not_contain": [],
        },
    },
    {
        "id": 24,
        "title": "Temporal — did the planner act before or after the disruption?",
        "question": (
            "The Antwerp port disruption affected several shipments. For each "
            "affected shipment, did the planner's expedite or rerouting "
            "decision happen BEFORE or AFTER the disruption was resolved? "
            "List each with the disruption dates, the decision date, and "
            "whether the response was proactive or reactive."
        ),
        "why_hard": (
            "Baseline must: (1) get disruption_events (Antwerp, dates), "
            "(2) get planner_decisions (dates per shipment), (3) get shipments "
            "to link them, (4) compare timestamps across 3 separate JSON "
            "payloads. Models often get temporal ordering wrong when dates "
            "come from different tool responses. Cognee's graph co-locates "
            "disruption→shipment→decision with timestamps."
        ),
        "expected": {
            "must_contain": ["Antwerp", "SHP-002", "proactive"],
            "must_not_contain": [],
        },
    },
    {
        "id": 25,
        "title": "Temporal — tracking gap detection",
        "question": (
            "For SHP-006, how many days passed between the last tracking "
            "event and the revised ETA? Is there a gap with no visibility? "
            "What should the planner do about it?"
        ),
        "why_hard": (
            "Baseline must: (1) get tracking events for SHP-006 (find latest "
            "timestamp), (2) get shipment SHP-006 (revised_eta), (3) compute "
            "the gap in days. Requires two tool calls and date math."
        ),
        "expected": {
            "must_contain": ["SHP-006"],
            "must_not_contain": [],
        },
    },
    # ── NEW: Negative reasoning questions ─────────────────────────────
    {
        "id": 26,
        "title": "Negative — no mitigation plan for SUPP-013",
        "question": (
            "Has anyone documented a mitigation plan or contingency decision "
            "for the Rhine Components Asia entity (SUPP-013)? Check planner "
            "decisions, feedback corrections, and disruption events. If "
            "nothing exists, flag this as a risk gap."
        ),
        "why_hard": (
            "Answer: NO — no planner decision, feedback correction, or "
            "disruption event references SUPP-013. The baseline must call "
            "3 tools, get results, and correctly conclude 'nothing found' "
            "rather than hallucinating a plan. Agents often say 'I found "
            "a mitigation plan' when they didn't, or give a vague answer."
        ),
        "expected": {
            "must_contain": ["SUPP-013", "no"],
            "must_not_contain": [],
        },
    },
    {
        "id": 27,
        "title": "Negative — orphan shipments without customer orders",
        "question": (
            "Are there any shipments that have no linked customer order? "
            "If so, list up to 10 of them with their PO, supplier, SKU, "
            "and current status. Why might these exist?"
        ),
        "why_hard": (
            "Requires: get all shipments, get all customer orders, find "
            "shipments whose IDs do NOT appear in any fulfillment_shipments "
            "field. Proving a negative across two large tables (250 shipments "
            "× 200 orders). The baseline must do a set-difference operation "
            "across two large JSON payloads."
        ),
        "expected": {
            "must_contain": ["SHP-1", "no", "customer order"],
            "must_not_contain": [],
        },
    },
    {
        "id": 28,
        "title": "Negative — missing feedback for specific shipments",
        "question": (
            "Has there been any feedback correction or root cause override "
            "for SHP-017? What about SHP-016? For each, say whether a "
            "correction exists and if so, what the corrected root cause is."
        ),
        "why_hard": (
            "The baseline must search feedback_corrections for both SHP-017 "
            "and SHP-016 and correctly report the absence for one and the "
            "presence for the other. Agents often say 'I couldn't find it' "
            "ambiguously or hallucinate a correction that doesn't exist."
        ),
        "expected": {
            "must_contain": ["SHP-017", "SHP-016"],
            "must_not_contain": [],
        },
    },
    # ── NEW: Policy deep-understanding questions ──────────────────────
    {
        "id": 29,
        "title": "Policy nuance — expedite approval threshold vs actual decision",
        "question": (
            "Our expedite policy defines a cost threshold above which VP "
            "approval is required instead of Operations Director approval. "
            "What is that threshold? Was the SHP-003 expedite within or "
            "above it? Who should have approved it, and what justified the "
            "approval given the Nexus blanket order at stake?"
        ),
        "why_hard": (
            "The policy states the Operations Director can approve up to "
            "$5,000; above that requires VP approval. SHP-003 cost $4,500, "
            "so Ops Director was sufficient. But the justification is "
            "nuanced: the $180,000 blanket order at risk (from customer "
            "brief) outweighs the $4,500 cost. The baseline must: "
            "(1) find and read the expedite policy (lookup_policy topic), "
            "(2) extract the $5,000 threshold, (3) find the SHP-003 "
            "decision cost ($4,500), (4) read the Nexus customer brief "
            "for the blanket order value. That's 2 policy docs + planner "
            "decisions + shipments — 4 sources with textual reasoning. "
            "Cognee's graph connects the policy text, decision, shipment, "
            "and customer brief in the same subgraph."
        ),
        "expected": {
            "must_contain": ["5,000", "4,500", "Operations Director", "180,000"],
            "must_not_contain": [],
        },
    },
    {
        "id": 30,
        "title": "Policy nuance — when NOT to expedite (3 exclusion criteria)",
        "question": (
            "Our expedite policy lists specific scenarios where expediting "
            "is NOT advisable. What are those scenarios? For each one, "
            "can you find a real shipment or order in our data that matches "
            "that exclusion? Give specific IDs."
        ),
        "why_hard": (
            "The policy lists 3 exclusion criteria: (1) commodity/P3 orders "
            "without penalty clauses, (2) shipments already too late to "
            "meet the deadline, (3) a backup shipment already booked and "
            "on time (e.g. SHP-008). The baseline must read the policy "
            "text, extract these rules, then search shipments, customer "
            "orders, and POs to find examples matching each rule. This "
            "requires interpreting nuanced policy language AND mapping it "
            "to data. The baseline typically reads the policy but doesn't "
            "cross-reference to find matching examples. Cognee's graph "
            "has the policy text connected to the entities it references "
            "(SHP-008, commodity SKUs, P3 customers)."
        ),
        "expected": {
            "must_contain": ["SHP-008", "P3", "commodity"],
            "must_not_contain": [],
        },
    },
    {
        "id": 31,
        "title": "Policy nuance — Q1 risk brief + escalation playbook interaction",
        "question": (
            "Our Q1 European Lane Risk Brief recommends pre-positioning "
            "buffer stock and adjusting lead times for SUPP-002. What "
            "specific lead time adjustment does it recommend? If a port "
            "strike is announced during Q1, which playbook should be "
            "activated, and what are the first 3 actions within 24 hours "
            "according to that playbook? Has any of this actually been "
            "done for current shipments?"
        ),
        "why_hard": (
            "The risk brief says plan for 21 days from SUPP-002 instead "
            "of the typical 14. If a port strike is announced, the "
            "European Lane Disruption Playbook should be activated. "
            "Its 24h actions: (1) identify all in-transit/booked shipments "
            "on affected lanes, (2) calculate penalty exposure, (3) "
            "evaluate backup from unaffected lane (e.g. SUPP-004 via "
            "Busan). The baseline must: read 2 policy documents "
            "(risk brief + escalation playbook), extract specific numbers "
            "and action items, then check planner_decisions and shipments "
            "to see if these were actually followed. Most agents return "
            "a vague summary of one document and miss the other. "
            "Cognee's graph connects both policy docs to the same "
            "supplier/lane/shipment entities."
        ),
        "expected": {
            "must_contain": ["21 days", "14", "SUPP-002", "SUPP-004", "Busan", "penalty"],
            "must_not_contain": [],
        },
    },
    {
        "id": 32,
        "title": "Policy nuance — DG carrier selection vs cost-based selection",
        "question": (
            "Our dangerous goods policy mentions a specific past incident "
            "where an agent recommended the wrong carrier for SKU-003 "
            "based on cost alone. What happened, which carrier should have "
            "been selected, and what is the policy's explicit rule about "
            "cost-based carrier selection for regulated materials? Also, "
            "which of our current Hamburg→LA carriers are actually DG "
            "approved according to carrier performance data?"
        ),
        "why_hard": (
            "The DG policy describes an incident where an agent recommended "
            "the 'cheapest carrier' for SKU-003 and was corrected — the "
            "carrier's DG approval must always be verified. For air, only "
            "DHL Express is approved. For sea, Maersk is designated for "
            "Hamburg routes. The baseline must: (1) read the DG policy "
            "(lookup_policy 'dangerous_goods'), (2) extract the incident "
            "narrative and the rule, (3) read carrier_performance for "
            "Hamburg→LA carriers and their dg_approved status. The policy "
            "keyword 'dangerous_goods' must be guessed correctly. Then "
            "the agent must connect the policy narrative to real carrier "
            "data. Cognee's graph has the policy text linked to SKU-003, "
            "carriers, and the DG approval status."
        ),
        "expected": {
            "must_contain": ["DHL Express", "Maersk", "UN3480", "dg_approved", "cost"],
            "must_not_contain": [],
        },
    },
    # ── Synonym-only questions (no CSV column value matches) ──────────
    {
        "id": 33,
        "title": "Synonym only — Asian semiconductor partner with worst delivery",
        "question": (
            "Which of our Asian semiconductor partners has the worst "
            "reliability score? What is their score, what products do they "
            "supply, and which customer orders depend on those products?"
        ),
        "why_hard": (
            "'Asian' must map to China, South Korea, Japan. 'semiconductor' "
            "does not appear in any CSV column — the agent must infer from "
            "SKU descriptions or categories. The correct answer is SUPP-010 "
            "(Seoul Micro Industries, South Korea, score 4.3). Baseline must "
            "try multiple country filters or dump all suppliers, then cross-"
            "reference SKUs and customer orders."
        ),
        "expected": {
            "must_contain": ["SUPP-010", "Seoul Micro", "4.3"],
            "must_not_contain": [],
        },
    },
    {
        "id": 34,
        "title": "Synonym only — Suez Canal closure exposure",
        "question": (
            "If the Suez Canal closes for 2 weeks starting tomorrow, which "
            "of our active shipments would be affected? What is the total "
            "daily penalty exposure from the customer orders those shipments "
            "fulfill, and what alternative carriers or routes exist in our "
            "carrier performance data?"
        ),
        "why_hard": (
            "No column says 'Suez Canal.' The agent must reason that "
            "Hamburg→LA via sea transits the Suez Canal, find all active "
            "Hamburg→LA sea shipments, trace them to customer orders for "
            "penalties, and check carrier_performance for alternatives. "
            "Pure geographic reasoning with no direct filter match."
        ),
        "expected": {
            "must_contain": ["Hamburg", "SKU-003", "penalty", "Maersk"],
            "must_not_contain": [],
        },
    },
    # ── Arithmetic questions (require calculations over many rows) ────
    {
        "id": 35,
        "title": "Arithmetic — weighted supplier reliability by PO count",
        "question": (
            "Which supplier has the most non-delivered purchase orders in "
            "our system right now? How many non-delivered POs do they have "
            "compared to the next-largest supplier? What is their reliability "
            "score and what does this mean for our risk exposure?"
        ),
        "why_hard": (
            "Must count non-delivered POs per supplier across 243 PO rows. "
            "SUPP-002 has the most (large number of AT_RISK/DELAYED POs). "
            "The baseline must filter purchase_orders by multiple statuses, "
            "group by supplier, count, and compare — hard to do accurately "
            "with in-context arithmetic over large JSON payloads."
        ),
        "expected": {
            "must_contain": ["SUPP-002", "Rhine Components", "3.2"],
            "must_not_contain": [],
        },
    },
    {
        "id": 36,
        "title": "Arithmetic — top customers by penalty exposure",
        "question": (
            "Which customer has the highest total daily penalty rate across "
            "all their active (non-delivered) orders? List the top 3 customers "
            "by total daily penalty, showing for each: customer name, number "
            "of active orders, and combined daily penalty in $/day."
        ),
        "why_hard": (
            "Must iterate over ~200 customer orders, filter non-delivered, "
            "sum penalty_per_day_usd per customer, sort, and take top 3. "
            "With distractor data the sums are large and error-prone for "
            "in-context arithmetic. Nexus Robotics should appear near the "
            "top given their $244/day penalty on CUST-ORD-1042."
        ),
        "expected": {
            "must_contain": ["Nexus Robotics", "penalty", "$/day"],
            "must_not_contain": [],
        },
    },
    # ── Contradictory data reconciliation ─────────────────────────────
    {
        "id": 37,
        "title": "Contradictory data — TMS status vs tracking events for SHP-006",
        "question": (
            "SHP-006 shows status 'DELAYED' in our TMS with a revised ETA. "
            "But what does the actual tracking event history say about its "
            "current location and progress? Compare the TMS record with the "
            "tracking events and tell me which source is more up-to-date. "
            "Has the planner acknowledged any discrepancy?"
        ),
        "why_hard": (
            "Must query shipments for the TMS view, tracking_events for "
            "SHP-006's event history, and planner_notes/feedback for any "
            "acknowledgment. The agent must reconcile two different data "
            "sources rather than just trusting the first one it reads."
        ),
        "expected": {
            "must_contain": ["SHP-006", "DELAYED", "tracking"],
            "must_not_contain": [],
        },
    },
    {
        "id": 38,
        "title": "Contradictory data — verify SHP-003 expedite was executed",
        "question": (
            "Planner notes say SHP-003 was expedited from sea to air via "
            "DHL Express. Does the actual shipment record in our TMS confirm "
            "this mode change? Cross-check the shipment data, tracking "
            "events, and carrier records. Was the expedite actually executed "
            "as planned?"
        ),
        "why_hard": (
            "Most agents just repeat the planner note without verifying. "
            "The baseline must check: (1) shipments.csv for SHP-003 carrier "
            "and freight_mode, (2) tracking_events for SHP-003, (3) carrier "
            "performance for DHL. The shipment record does show DHL Express "
            "and Air mode, so the expedite was executed — but the agent must "
            "prove it from data, not just echo the planner note."
        ),
        "expected": {
            "must_contain": ["SHP-003", "DHL Express", "Air"],
            "must_not_contain": [],
        },
    },
]


# ── Scoring ───────────────────────────────────────────────────────────


def score_answer(answer: str, expected: dict) -> dict:
    """Check if answer contains all required facts and no forbidden ones."""
    answer_lower = answer.lower()
    must = expected.get("must_contain", [])
    must_not = expected.get("must_not_contain", [])
    hits = [k for k in must if k.lower() in answer_lower]
    misses = [k for k in must if k.lower() not in answer_lower]
    false_pos = [k for k in must_not if k.lower() in answer_lower]
    score = len(hits) / max(len(must), 1)
    return {
        "score": round(score, 2),
        "hits": hits,
        "misses": misses,
        "false_positives": false_pos,
    }


def diagnose_agent(
    label: str,
    answer: str,
    score_result: dict | None,
    metrics: dict,
    why_hard: str,
) -> str:
    """Generate a human-readable explanation of why an agent scored the way it did."""
    if score_result is None:
        return ""

    lines = []
    pct = score_result["score"]
    misses = score_result["misses"]
    false_pos = score_result["false_positives"]

    if pct >= 1.0 and not false_pos:
        lines.append(f"{label}: PASS — all required facts present.")
        return "\n".join(lines)

    lines.append(f"{label}: FAIL — scored {pct:.0%}.")

    stop_reason = metrics.get("stop_reason", "")
    max_rounds = metrics.get("max_rounds", 0)
    llm_calls = metrics.get("llm_calls", 0)
    tool_calls = metrics.get("tool_calls", 0)
    total_chars = metrics.get("total_result_chars", 0)
    answer_stripped = answer.strip()

    reasons = []

    if stop_reason == "max_rounds_exhausted":
        reasons.append(
            f"Hit the {max_rounds}-round limit ({llm_calls} LLM calls, "
            f"{tool_calls} tool calls). The agent ran out of reasoning "
            f"rounds before it could synthesize a final answer."
        )

    if not answer_stripped:
        reasons.append(
            "Returned an empty answer — likely spent all rounds on tool "
            "calls and never produced a text response."
        )
    elif len(answer_stripped) < 80:
        reasons.append(
            f"Very short answer ({len(answer_stripped)} chars) — may have "
            "given up or asked a clarifying question instead of answering."
        )

    if total_chars > 100_000:
        reasons.append(
            f"Pulled {total_chars:,} chars of tool data into context. "
            "Large payloads can overwhelm the LLM, causing it to lose "
            "track of key facts or truncate its reasoning."
        )

    filter_errors = [
        t
        for t in metrics.get("tool_timings", [])
        if t.get("result_chars", 0) > 0 and t["result_chars"] < 80
    ]
    if filter_errors:
        names = [t["tool"] for t in filter_errors]
        reasons.append(
            f"Some tool calls returned very little data ({', '.join(names)}) — "
            "possibly wrong filters or the required filter was missing."
        )

    if misses:
        if tool_calls == 0:
            reasons.append(
                f"Made no tool calls — tried to answer from memory alone and missed: {misses}."
            )
        elif tool_calls <= 2 and len(misses) > 2:
            reasons.append(
                f"Only made {tool_calls} tool call(s) — insufficient data "
                f"gathering to cover all required facts. Missed: {misses}."
            )
        else:
            reasons.append(f"Missing required facts: {misses}.")

    if false_pos:
        reasons.append(f"Included incorrect/forbidden information: {false_pos}.")

    if not reasons:
        reasons.append(
            "The answer was produced but did not contain all expected keywords. "
            "The agent may have used different phrasing or missed details."
        )

    for i, r in enumerate(reasons, 1):
        lines.append(f"  {i}. {r}")

    return "\n".join(lines)


# ── Graph setup ───────────────────────────────────────────────────────


async def build_cognee_graph() -> dict:
    """Build the knowledge graph. Returns token usage metrics for the build."""
    cognee_agent._tracker.reset()
    build_start = time.perf_counter()

    _echo("Resetting Cognee data …")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    _echo("  Building graph from structured CSVs (deterministic) …")
    node_count = await build_graph_from_csvs()
    _echo(f"  {node_count} nodes created from CSVs.")

    text_sources = {
        "planner_notes": get_planner_notes_data(),
        "disruptions": get_disruption_alert(),
    }
    for name, text in text_sources.items():
        _echo(f"  Adding {name} ({len(text.split())} words) for LLM extraction …")
        await cognee.add(text, node_set=[name])

    paths = get_policy_document_paths()
    if paths:
        _echo(f"  Adding {len(paths)} policy documents …")
        await cognee.add(paths, node_set=["policies"])

    _echo("  Running cognify on text sources (planner notes, policies, disruptions) …")
    await cognee.cognify(graph_model=SupplyChainContext)

    user = await get_default_user()
    await create_triplet_embeddings(user=user, dataset="main_dataset", triplets_batch_size=50)

    build_ms = (time.perf_counter() - build_start) * 1000
    build_tokens = cognee_agent._tracker.snapshot()
    _echo(
        f"  Knowledge graph ready.  "
        f"Build cost: {build_tokens['llm_calls']} LLM calls, "
        f"{build_tokens['prompt_tokens']:,} prompt + "
        f"{build_tokens['completion_tokens']:,} completion = "
        f"{build_tokens['total_tokens']:,} tokens, "
        f"{build_ms:,.0f}ms\n"
    )
    return {**build_tokens, "build_ms": round(build_ms, 1)}


# ── Comparison runner ─────────────────────────────────────────────────


async def run_scenario(
    step_ids: list[int] | None = None,
    build_metrics: dict | None = None,
    summarize: bool = False,
    max_rounds: int = 8,
    cognee_search_type=None,
) -> None:
    if step_ids:
        id_set = set(step_ids)
        steps_to_run = [s for s in SCENARIO if s["id"] in id_set]
    else:
        steps_to_run = SCENARIO

    results = []

    for step in steps_to_run:
        _print_section(f"Step {step['id']} — {step['title']}")
        _echo(f"\nQuestion: {step['question']}\n")
        _echo(f"Why this is hard: {step['why_hard']}\n")

        expected = step.get("expected", {})

        _echo("--- BASELINE (tool-calling agent over source APIs) ---")
        b_answer, b_tools, b_metrics = await baseline_agent.run(
            step["question"], summarize=summarize, max_rounds=max_rounds
        )
        _echo(f"Tools called: {', '.join(b_tools) if b_tools else 'none'}")
        _echo(
            f"Performance: {b_metrics['llm_calls']} LLM calls, "
            f"{b_metrics['tool_calls']} tool calls, "
            f"{b_metrics['total_ms']:.0f}ms total, "
            f"{b_metrics['total_result_chars']:,} chars returned"
        )
        _echo(
            f"Tokens: {b_metrics['prompt_tokens']:,} prompt + "
            f"{b_metrics['completion_tokens']:,} completion = "
            f"{b_metrics['total_tokens']:,} total"
        )
        if b_metrics.get("context_summarized"):
            _echo("  (context was summarized mid-run — lossy compression applied)")
        for t in b_metrics["tool_timings"]:
            args_str = ", ".join(f"{k}={v!r}" for k, v in t["args"].items())
            _echo(f"  {t['tool']}({args_str}) -> {t['result_chars']:,} chars")
        _echo(f"Answer:\n{_wrap(b_answer)}\n")
        b_score = score_answer(b_answer, expected) if expected else None
        if b_score:
            _echo(
                f"Score: {b_score['score']:.0%} "
                f"(hits: {b_score['hits']}, misses: {b_score['misses']})"
            )
            if b_score["false_positives"]:
                _echo(f"  False positives: {b_score['false_positives']}")
        b_diagnosis = diagnose_agent("Baseline", b_answer, b_score, b_metrics, step["why_hard"])
        if b_diagnosis:
            _echo(b_diagnosis)

        _echo("\n--- COGNEE (knowledge graph agent) ---")
        c_t0 = time.perf_counter()
        run_kwargs = {}
        if cognee_search_type is not None:
            run_kwargs["query_type"] = cognee_search_type
        c_answer, c_tools, c_metrics = await cognee_agent.run(step["question"], **run_kwargs)
        c_ms = (time.perf_counter() - c_t0) * 1000
        _echo(f"Tools called: {', '.join(c_tools) if c_tools else 'none'}")
        _echo(
            f"Performance: {c_metrics['llm_calls']} LLM calls, 1 graph search, {c_ms:.0f}ms total"
        )
        _echo(
            f"Tokens: {c_metrics['prompt_tokens']:,} prompt + "
            f"{c_metrics['completion_tokens']:,} completion = "
            f"{c_metrics['total_tokens']:,} total"
        )
        _echo(f"Answer:\n{_wrap(c_answer)}\n")
        c_score = score_answer(c_answer, expected) if expected else None
        if c_score:
            _echo(
                f"Score: {c_score['score']:.0%} "
                f"(hits: {c_score['hits']}, misses: {c_score['misses']})"
            )
            if c_score["false_positives"]:
                _echo(f"  False positives: {c_score['false_positives']}")
        c_diagnosis = diagnose_agent("Cognee", c_answer, c_score, c_metrics, step["why_hard"])
        if c_diagnosis:
            _echo(c_diagnosis)

        results.append(
            {
                "id": step["id"],
                "title": step["title"],
                "baseline_score": b_score["score"] if b_score else None,
                "cognee_score": c_score["score"] if c_score else None,
                "baseline_misses": b_score["misses"] if b_score else [],
                "cognee_misses": c_score["misses"] if c_score else [],
                "baseline_ms": b_metrics["total_ms"],
                "cognee_ms": round(c_ms, 1),
                "baseline_tool_calls": b_metrics["tool_calls"],
                "baseline_llm_calls": b_metrics["llm_calls"],
                "baseline_chars": b_metrics["total_result_chars"],
                "baseline_tokens": b_metrics["total_tokens"],
                "cognee_tokens": c_metrics["total_tokens"],
                "cognee_llm_calls": c_metrics["llm_calls"],
            }
        )

    _print_section("Scoring Summary")
    header = (
        f"{'Step':<6} {'Title':<36} "
        f"{'B.Score':>7} {'C.Score':>7} "
        f"{'B.ms':>7} {'C.ms':>7} "
        f"{'B.Tok':>8} {'C.Tok':>8} "
        f"{'B.Tools':>7} {'B.Chars':>8}"
    )
    _echo(header)
    _echo("-" * len(header))
    b_total, c_total, count = 0.0, 0.0, 0
    b_ms_total, c_ms_total = 0.0, 0.0
    b_tok_total, c_tok_total = 0, 0
    for r in results:
        b_pct = f"{r['baseline_score']:.0%}" if r["baseline_score"] is not None else "N/A"
        c_pct = f"{r['cognee_score']:.0%}" if r["cognee_score"] is not None else "N/A"
        title = r["title"][:34]
        b_ms = f"{r.get('baseline_ms', 0):.0f}"
        c_ms = f"{r.get('cognee_ms', 0):.0f}"
        b_tok = f"{r.get('baseline_tokens', 0):,}"
        c_tok = f"{r.get('cognee_tokens', 0):,}"
        b_tc = str(r.get("baseline_tool_calls", 0))
        b_ch = f"{r.get('baseline_chars', 0):,}"
        _echo(
            f"{r['id']:<6} {title:<36} "
            f"{b_pct:>7} {c_pct:>7} "
            f"{b_ms:>7} {c_ms:>7} "
            f"{b_tok:>8} {c_tok:>8} "
            f"{b_tc:>7} {b_ch:>8}"
        )
        if r["baseline_score"] is not None:
            b_total += r["baseline_score"]
            count += 1
        if r["cognee_score"] is not None:
            c_total += r["cognee_score"]
        b_ms_total += r.get("baseline_ms", 0)
        c_ms_total += r.get("cognee_ms", 0)
        b_tok_total += r.get("baseline_tokens", 0)
        c_tok_total += r.get("cognee_tokens", 0)
    _echo("-" * len(header))
    if count > 0:
        _echo(
            f"{'AVG':<6} {'':<36} "
            f"{b_total / count:>6.0%} {c_total / count:>7.0%} "
            f"{b_ms_total / count:>6.0f} {c_ms_total / count:>7.0f} "
            f"{b_tok_total // count:>8,} {c_tok_total // count:>8,}"
        )
        _echo(
            f"{'TOTAL':<6} {'':<36} "
            f"{'':>7} {'':>7} "
            f"{b_ms_total:>6.0f} {c_ms_total:>7.0f} "
            f"{b_tok_total:>8,} {c_tok_total:>8,}"
        )
    _echo("")
    if build_metrics:
        _echo(
            f"Cognee graph build (one-time setup): "
            f"{build_metrics['llm_calls']} LLM calls, "
            f"{build_metrics['total_tokens']:,} tokens "
            f"({build_metrics['prompt_tokens']:,} prompt + "
            f"{build_metrics['completion_tokens']:,} completion), "
            f"{build_metrics['build_ms']:,.0f}ms"
        )
        _echo(
            f"Cognee total cost (build + queries): "
            f"{build_metrics['total_tokens'] + c_tok_total:,} tokens, "
            f"{build_metrics['build_ms'] + c_ms_total:,.0f}ms"
        )
        _echo("")
    _echo(
        f"The baseline agent has {len(baseline_agent.TOOL_FUNCTIONS)} tools "
        f"and must decide which source APIs to query with correct filters.\n"
        "It receives JSON payloads and must manually chain IDs across systems.\n"
        "The Cognee agent has 1 tool — the knowledge graph — which already\n"
        "connects all entities and relationships across all sources.\n"
    )


# ── Main ──────────────────────────────────────────────────────────────


def _parse_args():
    """Parse CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: Baseline vs Cognee agent."
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip graph build phase.")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Enable context summarization in baseline agent.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=8,
        help="Maximum LLM rounds for the baseline agent (default: 8).",
    )

    parser.add_argument(
        "--cognee-search-type",
        default="GRAPH_COMPLETION",
        help=(
            "Cognee SearchType for the cognee agent "
            "(default: GRAPH_COMPLETION). "
            "Try GRAPH_COMPLETION_COT for chain-of-thought."
        ),
    )

    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument("--step", type=int, help="Run a single step by ID.")
    step_group.add_argument(
        "--steps",
        nargs="+",
        help="Run specific steps (comma or space separated, e.g. --steps 3,9,11 or --steps 3 9 11).",
    )

    args = parser.parse_args()

    step_ids = None
    if args.step is not None:
        step_ids = [args.step]
    elif args.steps is not None:
        step_ids = []
        for token in args.steps:
            for part in token.split(","):
                part = part.strip()
                if part.isdigit():
                    step_ids.append(int(part))
        step_ids = step_ids if step_ids else None

    from cognee import SearchType

    try:
        cognee_search_type = SearchType[args.cognee_search_type]
    except KeyError:
        valid = ", ".join(t.name for t in SearchType)
        parser.error(f"Invalid search type '{args.cognee_search_type}'. Valid: {valid}")

    return step_ids, args.skip_build, args.summarize, args.max_rounds, cognee_search_type


async def main() -> None:
    global _log_file
    _log_file = open(LOG_PATH, "w", encoding="utf-8")

    step_ids, skip_build, summarize, max_rounds, cognee_search_type = _parse_args()

    try:
        _echo(f"--- Comparison run started at {datetime.now().isoformat()} ---\n")
        _echo(
            f"Settings: max_rounds={max_rounds}, summarize={summarize}, "
            f"cognee_search_type={cognee_search_type.name}\n"
        )

        build_metrics = None
        if not skip_build:
            _print_section("Phase 1 — Build the Cognee knowledge graph")
            build_metrics = await build_cognee_graph()

        if step_ids:
            label = f"steps {','.join(str(s) for s in step_ids)}"
        else:
            label = f"{len(SCENARIO)}-step scenario"
        _print_section(f"Phase 2 — Run {label} through both agents")
        await run_scenario(
            step_ids=step_ids,
            build_metrics=build_metrics,
            summarize=summarize,
            max_rounds=max_rounds,
            cognee_search_type=cognee_search_type,
        )

        _echo(f"\nFull log saved to: {LOG_PATH}")
    finally:
        if _log_file:
            _log_file.close()
            _log_file = None


if __name__ == "__main__":
    asyncio.run(main())
