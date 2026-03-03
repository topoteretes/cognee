"""Baseline tool-calling agent — each tool queries a source system API.

The LLM orchestrator must decide which tools to call, apply the right
filters, combine data from multiple systems, and synthesize an answer.
No knowledge graph, no persistent memory.

Large tables (shipments, purchase orders, customer orders, tracking events,
feedback corrections) require at least one filter parameter — they do not
allow unfiltered dumps, mirroring real enterprise API behaviour.
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional

import litellm
from cognee.infrastructure.llm.config import get_llm_config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")

SYSTEM_PROMPT = (
    "You are a supply chain planner assistant at ACME Electronics. "
    "You have access to tools that query individual source systems "
    "(ERP, TMS, OMS, planning logs, etc.). Each tool may require filter "
    "parameters — read the tool descriptions carefully. "
    "If a question requires data from multiple systems, call multiple tools. "
    "Combine and cross-reference the results to synthesize a clear answer."
)


def _read_csv_raw(filename: str) -> str:
    """Read a small CSV file and return its full text (for small reference tables)."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return f.read()


def _read_csv_filtered(
    filename: str,
    filters: Optional[dict] = None,
    require_filter: bool = False,
) -> str:
    """Read a CSV, apply case-insensitive substring filters, return JSON.

    If require_filter is True and no filters are provided, returns an error
    message telling the caller to supply at least one filter.
    """
    if require_filter and not filters:
        path = os.path.join(DATA_DIR, filename)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
        return (
            f"Error: This endpoint requires at least one filter parameter. "
            f"Available columns to filter on: {columns}"
        )

    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if filters:
                if not all(v.lower() in row.get(k, "").lower() for k, v in filters.items() if v):
                    continue
            rows.append(dict(row))

    if not rows:
        return json.dumps({"results": [], "count": 0})
    return json.dumps({"results": rows, "count": len(rows)}, indent=2)


def _read_doc(filename: str) -> str:
    path = os.path.join(DOCS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── Tool implementations ──────────────────────────────────────────────


def lookup_suppliers(
    supplier_id: Optional[str] = None, name: Optional[str] = None, country: Optional[str] = None
) -> str:
    filters = {}
    if supplier_id:
        filters["supplier_id"] = supplier_id
    if name:
        filters["name"] = name
    if country:
        filters["country"] = country
    return _read_csv_filtered("suppliers.csv", filters or None)


def lookup_skus(
    sku_id: Optional[str] = None, category: Optional[str] = None, criticality: Optional[str] = None
) -> str:
    filters = {}
    if sku_id:
        filters["sku_id"] = sku_id
    if category:
        filters["category"] = category
    if criticality:
        filters["criticality"] = criticality
    return _read_csv_filtered("skus.csv", filters or None)


def lookup_purchase_orders(
    po_number: Optional[str] = None,
    supplier_id: Optional[str] = None,
    sku_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    filters = {}
    if po_number:
        filters["po_number"] = po_number
    if supplier_id:
        filters["supplier_id"] = supplier_id
    if sku_id:
        filters["sku_id"] = sku_id
    if status:
        filters["status"] = status
    return _read_csv_filtered("purchase_orders.csv", filters, require_filter=True)


def lookup_shipments(
    shipment_id: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    sku: Optional[str] = None,
    carrier: Optional[str] = None,
    status: Optional[str] = None,
    po_reference: Optional[str] = None,
) -> str:
    filters = {}
    if shipment_id:
        filters["shipment_id"] = shipment_id
    if origin:
        filters["origin_port"] = origin
    if destination:
        filters["destination_port"] = destination
    if sku:
        filters["sku_id"] = sku
    if carrier:
        filters["carrier"] = carrier
    if status:
        filters["status"] = status
    if po_reference:
        filters["po_reference"] = po_reference
    return _read_csv_filtered("shipments.csv", filters, require_filter=True)


def lookup_tracking_events(
    shipment_id: Optional[str] = None, event_type: Optional[str] = None
) -> str:
    filters = {}
    if shipment_id:
        filters["shipment_id"] = shipment_id
    if event_type:
        filters["event_type"] = event_type
    return _read_csv_filtered("tracking_events.csv", filters, require_filter=True)


def lookup_customer_orders(
    order_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    sku: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    filters = {}
    if order_id:
        filters["order_id"] = order_id
    if customer_name:
        filters["customer_name"] = customer_name
    if sku:
        filters["required_skus"] = sku
    if status:
        filters["status"] = status
    return _read_csv_filtered("customer_orders.csv", filters, require_filter=True)


def lookup_planner_notes(
    shipment: Optional[str] = None, decision_maker: Optional[str] = None
) -> str:
    """Return planner war-room notes and decision logs (not feedback corrections)."""
    notes = _read_csv_raw("planner_notes.csv")

    filters = {}
    if shipment:
        filters["applies_to_shipment"] = shipment
    if decision_maker:
        filters["decision_maker"] = decision_maker
    decisions = _read_csv_filtered("planner_decisions.csv", filters or None)

    return notes + "\n\n--- PLANNER DECISIONS ---\n" + decisions


def lookup_feedback_corrections(
    shipment: Optional[str] = None, decision: Optional[str] = None
) -> str:
    """Return feedback corrections from the quality review system."""
    filters = {}
    if shipment:
        filters["related_shipment"] = shipment
    if decision:
        filters["related_decision"] = decision
    return _read_csv_filtered("feedback_corrections.csv", filters or None)


def lookup_disruption_events(
    event_type: Optional[str] = None, severity: Optional[str] = None
) -> str:
    filters = {}
    if event_type:
        filters["event_type"] = event_type
    if severity:
        filters["severity"] = severity
    return _read_csv_filtered("disruption_events.csv", filters or None)


def lookup_carrier_performance(carrier: Optional[str] = None, lane: Optional[str] = None) -> str:
    filters = {}
    if carrier:
        filters["carrier_name"] = carrier
    if lane:
        filters["lane"] = lane
    return _read_csv_filtered("carrier_performance.csv", filters or None)


def lookup_policy(topic: str) -> str:
    if not os.path.isdir(DOCS_DIR):
        return "No policy documents available."
    results = []
    for name in sorted(os.listdir(DOCS_DIR)):
        if name.endswith(".txt") and topic.lower().replace(" ", "_") in name.lower():
            results.append(f"--- {name} ---\n{_read_doc(name)}")
    if not results:
        available = [n for n in os.listdir(DOCS_DIR) if n.endswith(".txt")]
        return f"No policy matching '{topic}'. Available: {available}"
    return "\n\n".join(results)


# ── Tool registry ────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "lookup_suppliers": lookup_suppliers,
    "lookup_skus": lookup_skus,
    "lookup_purchase_orders": lookup_purchase_orders,
    "lookup_shipments": lookup_shipments,
    "lookup_tracking_events": lookup_tracking_events,
    "lookup_customer_orders": lookup_customer_orders,
    "lookup_planner_notes": lookup_planner_notes,
    "lookup_feedback_corrections": lookup_feedback_corrections,
    "lookup_disruption_events": lookup_disruption_events,
    "lookup_carrier_performance": lookup_carrier_performance,
    "lookup_policy": lookup_policy,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_suppliers",
            "description": "Query the supplier master list from ERP. Returns supplier IDs, names, countries, tiers, reliability scores, certifications. Filters are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "supplier_id": {
                        "type": "string",
                        "description": "Filter by supplier ID (e.g. 'SUPP-002').",
                    },
                    "name": {
                        "type": "string",
                        "description": "Filter by supplier name (substring match).",
                    },
                    "country": {
                        "type": "string",
                        "description": "Filter by country (substring match).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_skus",
            "description": "Query the SKU catalog from ERP. Returns SKU IDs, descriptions, criticality, sourcing type, BOM components, safety stock. Filters are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku_id": {
                        "type": "string",
                        "description": "Filter by SKU ID (e.g. 'SKU-003').",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (substring match).",
                    },
                    "criticality": {
                        "type": "string",
                        "description": "Filter by criticality level.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_purchase_orders",
            "description": "Query purchase orders from ERP. At least one filter is required. Returns PO numbers, supplier, SKU, quantities, status, linked shipments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "po_number": {
                        "type": "string",
                        "description": "Filter by PO number (e.g. 'PO-1001').",
                    },
                    "supplier_id": {
                        "type": "string",
                        "description": "Filter by supplier ID.",
                    },
                    "sku_id": {
                        "type": "string",
                        "description": "Filter by SKU ID.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status (e.g. 'open', 'shipped', 'delivered').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_shipments",
            "description": "Query shipment records from the TMS. At least one filter is required. Returns shipment IDs, carriers, routes, ETAs, delays, PO references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment_id": {
                        "type": "string",
                        "description": "Filter by shipment ID (e.g. 'SHP-006').",
                    },
                    "origin": {
                        "type": "string",
                        "description": "Filter by origin port (substring match, e.g. 'Hamburg').",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Filter by destination port (substring match, e.g. 'Los Angeles').",
                    },
                    "sku": {
                        "type": "string",
                        "description": "Filter by SKU ID.",
                    },
                    "carrier": {
                        "type": "string",
                        "description": "Filter by carrier name (substring match).",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by shipment status.",
                    },
                    "po_reference": {
                        "type": "string",
                        "description": "Filter by purchase order reference.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_tracking_events",
            "description": "Query tracking event logs from TMS. At least one filter is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment_id": {
                        "type": "string",
                        "description": "Filter by shipment ID (e.g. 'SHP-006').",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (e.g. 'departure', 'arrival', 'delay').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_customer_orders",
            "description": "Query customer orders from the order management system. At least one filter is required. Returns order IDs, customer names, required SKUs, penalty clauses, fulfillment shipments, status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Filter by order ID (e.g. 'CUST-ORD-1042').",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Filter by customer name (substring match).",
                    },
                    "sku": {
                        "type": "string",
                        "description": "Filter by required SKU (substring match in required_skus field).",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by order status.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_planner_notes",
            "description": "Query planner war-room notes and decision logs from the planning system. Optionally filter decisions by shipment or decision maker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment": {
                        "type": "string",
                        "description": "Filter decisions by shipment ID.",
                    },
                    "decision_maker": {
                        "type": "string",
                        "description": "Filter decisions by decision maker name.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_feedback_corrections",
            "description": "Query the quality review system for feedback corrections — cases where an original system answer was found to be wrong and corrected by a human reviewer. Useful for finding the real root cause of issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment": {
                        "type": "string",
                        "description": "Filter by related shipment ID.",
                    },
                    "decision": {
                        "type": "string",
                        "description": "Filter by related decision ID.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_disruption_events",
            "description": "Query disruption event alerts (port strikes, weather, supplier issues). Filters are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (e.g. 'port_strike', 'weather').",
                    },
                    "severity": {
                        "type": "string",
                        "description": "Filter by severity level.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_carrier_performance",
            "description": "Query carrier performance metrics — OTD rates, transit times, demurrage, DG approval, responsiveness. Filters are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "carrier": {
                        "type": "string",
                        "description": "Filter by carrier name (substring match).",
                    },
                    "lane": {
                        "type": "string",
                        "description": "Filter by trade lane (substring match, e.g. 'Hamburg-LA').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Search company policy/procedure documents by topic keyword (e.g. 'expedite', 'sole_source', 'dangerous_goods').",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic keyword to search policies for.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
]

CONTEXT_CHAR_LIMIT = 30_000

SUMMARIZE_PROMPT = (
    "Summarize the following tool results into a concise list of key facts. "
    "Keep all IDs, numbers, dates, and entity names. Drop raw JSON structure."
)


def _total_message_chars(messages: List[Dict]) -> int:
    """Sum the character length of all message content."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


async def _compress_context(messages: List[Dict], cfg) -> tuple[List[Dict], int, int]:
    """Replace tool/assistant messages with a single LLM-generated summary.

    Returns (new_messages, prompt_tokens_used, completion_tokens_used).
    """
    keep = []
    tool_texts = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ("system", "user") and "Summary of prior tool results" not in msg.get(
            "content", ""
        ):
            keep.append(msg)
        elif role == "tool":
            tool_texts.append(msg.get("content", ""))
        # assistant messages with tool_calls are dropped (stale after summary)

    combined = "\n---\n".join(tool_texts)
    response = await litellm.acompletion(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": combined},
        ],
        api_key=cfg.llm_api_key,
        api_base=cfg.llm_endpoint if cfg.llm_endpoint else None,
    )

    usage = getattr(response, "usage", None)
    p_tok = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    c_tok = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0

    summary = response.choices[0].message.content or ""
    keep.append(
        {
            "role": "user",
            "content": f"Summary of prior tool results:\n{summary}",
        }
    )
    return keep, p_tok, c_tok


# ── Agent loop ────────────────────────────────────────────────────────


async def run(
    question: str,
    history: Optional[List[Dict]] = None,
    summarize: bool = False,
    max_rounds: int = 6,
) -> tuple[str, List[str], dict]:
    """Run the baseline tool-calling agent.

    Returns (answer, list_of_tools_called, metrics).
    """
    cfg = get_llm_config()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    tools_called = []
    tool_timings = []
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    context_summarized = False
    total_start = time.perf_counter()

    def _build_metrics(stop_reason: str = "completed"):
        return {
            "total_ms": round((time.perf_counter() - total_start) * 1000, 1),
            "llm_calls": llm_calls,
            "tool_calls": len(tools_called),
            "tool_timings": tool_timings,
            "total_result_chars": sum(t["result_chars"] for t in tool_timings),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "max_rounds": max_rounds,
            "stop_reason": stop_reason,
            "context_summarized": context_summarized,
        }

    for _ in range(max_rounds):
        if summarize and _total_message_chars(messages) > CONTEXT_CHAR_LIMIT:
            messages, s_pt, s_ct = await _compress_context(messages, cfg)
            llm_calls += 1
            prompt_tokens += s_pt
            completion_tokens += s_ct
            context_summarized = True

        llm_calls += 1
        response = await litellm.acompletion(
            model=cfg.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            api_key=cfg.llm_api_key,
            api_base=cfg.llm_endpoint if cfg.llm_endpoint else None,
        )

        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" or (
            choice.message.tool_calls and len(choice.message.tool_calls) > 0
        ):
            messages.append(choice.message.model_dump())

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tools_called.append(fn_name)

                fn = TOOL_FUNCTIONS.get(fn_name)
                t0 = time.perf_counter()
                if fn:
                    result = fn(**fn_args)
                else:
                    result = f"Unknown tool: {fn_name}"
                elapsed_ms = (time.perf_counter() - t0) * 1000

                tool_timings.append(
                    {
                        "tool": fn_name,
                        "args": fn_args,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "result_chars": len(result),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            return choice.message.content or "", tools_called, _build_metrics("completed")

    last = response.choices[0].message.content or ""
    return last, tools_called, _build_metrics("max_rounds_exhausted")
