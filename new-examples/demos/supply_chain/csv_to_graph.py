"""Build the knowledge graph deterministically from CSV files.

Instead of converting CSVs to prose and relying on LLM extraction (which
drops fields and merges entities at scale), this module reads each CSV row
and creates typed DataPoint objects with proper cross-references.

Usage (from demo.py or standalone)::

    from csv_to_graph import build_graph_from_csvs
    await build_graph_from_csvs()
"""

from __future__ import annotations

import csv
import os
import uuid
from typing import Dict, List

from cognee.tasks.storage.add_data_points import add_data_points

from models import (
    CarrierLane,
    CustomerOrder,
    Decision,
    Feedback,
    Outcome,
    PurchaseOrder,
    Shipment,
    Site,
    SKU,
    Supplier,
    TrackingEvent,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _id(kind: str, key: str) -> uuid.UUID:
    """Deterministic UUID so the same CSV row always produces the same node."""
    return uuid.uuid5(NS, f"{kind}:{key}")


def _read(filename: str) -> List[Dict[str, str]]:
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _str(val, default: str = "") -> str:
    """Coerce None or missing CSV values to a safe string."""
    if val is None:
        return default
    return str(val)


def _float(val, default: float = 0.0) -> float:
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError, AttributeError):
        return default


def _int(val, default: int = 0) -> int:
    try:
        return int(float(str(val).replace("$", "").replace(",", "").strip()))
    except (ValueError, TypeError, AttributeError):
        return default


# ── Build helpers for each entity type ────────────────────────────────


def _build_suppliers(rows: List[Dict[str, str]]) -> Dict[str, Supplier]:
    out: Dict[str, Supplier] = {}
    for r in rows:
        sid = _str(r.get("supplier_id"))
        if not sid:
            continue
        out[sid] = Supplier(
            id=_id("Supplier", sid),
            supplier_id=sid,
            name=_str(r.get("name"), "n/a"),
            country=_str(r.get("country")),
            tier=_str(r.get("tier")),
            certifications=_str(r.get("certifications")),
            reliability_score=_float(r.get("reliability_score")),
            avg_lead_time_days=_int(r.get("avg_lead_time_days")),
            payment_terms=_str(r.get("payment_terms")),
            preferred_freight=_str(r.get("preferred_freight")),
            notes=_str(r.get("notes")),
        )
    return out


def _build_sites(rows: List[Dict[str, str]]) -> Dict[str, Site]:
    out: Dict[str, Site] = {}
    for r in rows:
        sid = _str(r.get("site_id"))
        if not sid:
            continue
        out[sid] = Site(
            id=_id("Site", sid),
            site_id=sid,
            name=_str(r.get("name"), "n/a"),
            region=_str(r.get("region")),
            city=_str(r.get("city")),
        )
    return out


def _build_skus(
    rows: List[Dict[str, str]],
    suppliers: Dict[str, Supplier],
) -> Dict[str, SKU]:
    out: Dict[str, SKU] = {}

    for r in rows:
        sid = _str(r.get("sku_id"))
        if not sid:
            continue
        supplied_by: List[Supplier] = []
        primary = _str(r.get("primary_supplier"))
        if primary and primary in suppliers:
            supplied_by.append(suppliers[primary])
        secondary = _str(r.get("secondary_supplier"))
        if secondary and secondary in suppliers:
            supplied_by.append(suppliers[secondary])

        safety_parts = []
        for site_key, label in [
            ("safety_stock_site_east", "SITE-EAST"),
            ("safety_stock_site_west", "SITE-WEST"),
            ("safety_stock_site_dc", "SITE-DC"),
        ]:
            val = _str(r.get(site_key))
            if val:
                safety_parts.append(f"{label}: {val}")
        safety_stock = "; ".join(safety_parts) if safety_parts else ""

        out[sid] = SKU(
            id=_id("SKU", sid),
            sku_id=sid,
            description=_str(r.get("description")),
            category=_str(r.get("category")),
            criticality=_str(r.get("criticality")),
            unit_cost=_float(r.get("unit_cost_usd")),
            sourcing_type=_str(r.get("sourcing_type")),
            special_handling=_str(r.get("special_handling")),
            safety_stock=safety_stock,
            supplied_by=supplied_by or None,
            components=None,
        )

    # Wire BOM components (second pass so all SKUs exist)
    for r in rows:
        bom_str = _str(r.get("bom_components"))
        if not bom_str:
            continue
        sku_node = out[r["sku_id"]]
        components: List[SKU] = []
        for part in bom_str.split(";"):
            part = part.strip()
            # e.g. "2×SKU-003" or "SKU-003"
            if "×" in part:
                comp_id = part.split("×", 1)[1].strip()
            elif "x" in part.lower():
                comp_id = part.split("x", 1)[1].strip()
                if not comp_id.startswith("SKU"):
                    comp_id = part.split("X", 1)[1].strip()
            else:
                comp_id = part
            if comp_id in out:
                components.append(out[comp_id])
        if components:
            sku_node.components = components

    return out


def _build_purchase_orders(
    rows: List[Dict[str, str]],
    suppliers: Dict[str, Supplier],
    skus: Dict[str, SKU],
    sites: Dict[str, Site],
) -> Dict[str, PurchaseOrder]:
    out: Dict[str, PurchaseOrder] = {}
    for r in rows:
        po_num = _str(r.get("po_number"))
        if not po_num:
            continue
        out[po_num] = PurchaseOrder(
            id=_id("PurchaseOrder", po_num),
            po_number=po_num,
            status=_str(r.get("status")),
            quantity=_int(r.get("quantity")),
            incoterm=_str(r.get("incoterm")),
            due_date=_str(r.get("due_date")),
            sku=skus.get(_str(r.get("sku_id"))),
            supplier=suppliers.get(_str(r.get("supplier_id"))),
            destination=sites.get(_str(r.get("destination_site"))),
        )
    return out


def _build_shipments(
    rows: List[Dict[str, str]],
    pos: Dict[str, PurchaseOrder],
    sites: Dict[str, Site],
) -> Dict[str, Shipment]:
    out: Dict[str, Shipment] = {}
    for r in rows:
        sid = _str(r.get("shipment_id"))
        if not sid:
            continue
        po_ref = _str(r.get("po_reference"))
        out[sid] = Shipment(
            id=_id("Shipment", sid),
            shipment_id=sid,
            carrier=_str(r.get("carrier")),
            status=_str(r.get("status")),
            origin_port=_str(r.get("origin_port")),
            destination_port=_str(r.get("destination_port")),
            freight_mode=_str(r.get("freight_mode")),
            expected_delivery=_str(r.get("original_eta") or r.get("expected_delivery")),
            actual_delivery=_str(r.get("actual_delivery")),
            total_delay_days=_int(r.get("total_delay_days")),
            notes=_str(r.get("notes")),
            purchase_order=pos.get(po_ref),
        )
    return out


def _build_tracking_events(
    rows: List[Dict[str, str]],
    shipments: Dict[str, Shipment],
) -> List[TrackingEvent]:
    out: List[TrackingEvent] = []
    for r in rows:
        eid = _str(r.get("event_id"))
        if not eid:
            continue
        out.append(
            TrackingEvent(
                id=_id("TrackingEvent", eid),
                event_type=_str(r.get("event_type"), "n/a"),
                timestamp=_str(r.get("timestamp"), "n/a"),
                location=_str(r.get("location")),
                port_code=_str(r.get("port_code")),
                delay_minutes=_int(r.get("delay_minutes")),
                delay_reason=_str(r.get("delay_reason_code")),
                details=_str(r.get("details")),
                shipment=shipments.get(_str(r.get("shipment_id"))),
            )
        )
    return out


def _build_carrier_lanes(rows: List[Dict[str, str]]) -> List[CarrierLane]:
    out: List[CarrierLane] = []
    for r in rows:
        carrier = _str(r.get("carrier_name"))
        lane = _str(r.get("lane"))
        period = _str(r.get("period"))
        if not carrier:
            continue
        key = f"{carrier}|{lane}|{period}"

        notes = _str(r.get("notes"))
        dg = _str(r.get("dg_approved"), "No")
        dg_flag = "DG-approved (UN3480)" if dg in ("Yes", "True") else "NOT DG-approved"
        notes = f"{dg_flag}. {notes}" if notes else dg_flag

        out.append(
            CarrierLane(
                id=_id("CarrierLane", key),
                carrier_name=carrier,
                lane=lane,
                period=period,
                total_shipments=_int(r.get("total_shipments")),
                otd_percent=_float(r.get("otd_percent")),
                avg_transit_days=_float(r.get("avg_transit_days")),
                congestion_events=_int(r.get("congestion_incidents")),
                demurrage_claims=_int(r.get("demurrage_claims_usd")),
                rate_info=f"${_str(r.get('rate_per_unit'), '0')}/{_str(r.get('rate_unit'), 'unit')}",
                notes=notes,
            )
        )
    return out


def _build_customer_orders(
    rows: List[Dict[str, str]],
    skus: Dict[str, SKU],
    sites: Dict[str, Site],
    shipments: Dict[str, Shipment],
) -> Dict[str, CustomerOrder]:
    out: Dict[str, CustomerOrder] = {}
    for r in rows:
        oid = _str(r.get("order_id"))
        if not oid:
            continue

        req_sku_str = _str(r.get("required_skus"))
        req_sku_ids = [s.strip() for s in req_sku_str.split(",") if s.strip()]
        required_skus = [skus[s] for s in req_sku_ids if s in skus] or None

        shp_str = _str(r.get("fulfillment_shipments"))
        shp_ids = [s.strip() for s in shp_str.split(",") if s.strip()]
        fulfilled_by = [shipments[s] for s in shp_ids if s in shipments] or None

        penalty_day = _float(r.get("penalty_per_day_usd"))
        penalty_pct = _float(r.get("penalty_percent_per_day"))
        penalty_cap = _int(r.get("penalty_cap_days"))
        status = _str(r.get("status"))
        penalty_details = f"Status: {status}." if status else ""
        if penalty_day > 0:
            penalty_details += (
                f" {penalty_pct}% per day = ${penalty_day:.0f}/day, cap {penalty_cap} days"
            )

        out[oid] = CustomerOrder(
            id=_id("CustomerOrder", oid),
            order_id=oid,
            customer_name=_str(r.get("customer_name")),
            priority=_str(r.get("account_tier")),
            order_value=_float(r.get("order_value_usd")),
            delivery_deadline=_str(r.get("delivery_deadline")),
            penalty_per_day=penalty_day,
            penalty_percent_per_day=penalty_pct,
            penalty_cap_days=penalty_cap,
            penalty_details=penalty_details,
            relationship_notes=_str(r.get("relationship_notes")),
            required_skus=required_skus,
            destination_site=sites.get(_str(r.get("destination_site"))),
            fulfilled_by=fulfilled_by,
        )
    return out


def _build_decisions(
    rows: List[Dict[str, str]],
    shipments: Dict[str, Shipment],
    pos: Dict[str, PurchaseOrder],
) -> Dict[str, Decision]:
    out: Dict[str, Decision] = {}
    for r in rows:
        did = _str(r.get("decision_id"))
        if not did:
            continue

        shp_ref = _str(r.get("applies_to_shipment"))
        applies_to_shp = None
        if shp_ref and shp_ref != "N/A":
            first_shp = shp_ref.split(",")[0].strip()
            applies_to_shp = shipments.get(first_shp)

        po_ref = _str(r.get("applies_to_po"))
        applies_to_po = None
        if po_ref and po_ref != "N/A":
            first_po = po_ref.split(",")[0].strip()
            applies_to_po = pos.get(first_po)

        rationale = _str(r.get("rationale"))
        lesson = _str(r.get("lessons_learned"))
        if lesson:
            rationale = f"{rationale} Lesson: {lesson}" if rationale else lesson

        out[did] = Decision(
            id=_id("Decision", did),
            decision_date=_str(r.get("decision_date")),
            action_taken=_str(r.get("action_taken"), "n/a"),
            rationale=rationale,
            cost_impact=_str(r.get("cost_impact_usd")),
            approved_by=_str(r.get("decision_maker")),
            applies_to=applies_to_shp,
            applies_to_po=applies_to_po,
        )
    return out


def _build_outcomes(
    rows: List[Dict[str, str]],
    decisions: Dict[str, Decision],
) -> List[Outcome]:
    """Build Outcome nodes from planner_decisions.csv (outcome columns)."""
    out: List[Outcome] = []
    for r in rows:
        outcome_date = _str(r.get("outcome_date"))
        outcome_result = _str(r.get("outcome_result"))
        if not outcome_date and not outcome_result:
            continue
        did = _str(r.get("decision_id"))
        if not did:
            continue
        out.append(
            Outcome(
                id=_id("Outcome", did),
                outcome_date=outcome_date,
                metric=_str(r.get("outcome_metric")),
                result_description=outcome_result or "n/a",
                value_before=_str(r.get("value_before")),
                value_after=_str(r.get("value_after")),
                decision=decisions.get(did),
            )
        )
    return out


def _build_feedback(
    rows: List[Dict[str, str]],
    decisions: Dict[str, Decision],
    shipments: Dict[str, Shipment],
) -> List[Feedback]:
    out: List[Feedback] = []
    for r in rows:
        fid = _str(r.get("feedback_id"))
        if not fid:
            continue

        related_decision = _str(r.get("related_decision"))
        corrects = None
        if related_decision:
            for d in decisions.values():
                if related_decision.lower() in d.action_taken.lower():
                    corrects = d
                    break

        related_shp_id = _str(r.get("related_shipment"))
        related_shp = None
        if related_shp_id and related_shp_id != "N/A":
            related_shp = shipments.get(related_shp_id.strip())

        original_query = _str(r.get("original_query"))
        original_answer = _str(r.get("original_answer"))
        if original_query:
            original_answer = f"Q: {original_query} A: {original_answer}"

        out.append(
            Feedback(
                id=_id("Feedback", fid),
                feedback_date=_str(r.get("feedback_date")),
                rating=_int(r.get("rating")),
                original_answer=original_answer,
                correction=_str(r.get("corrected_answer"), "n/a"),
                source_context=_str(r.get("root_cause_of_error")),
                corrects=corrects,
                related_shipment=related_shp,
            )
        )
    return out


# ── Main entry point ──────────────────────────────────────────────────


async def build_graph_from_csvs() -> int:
    """Read all structured CSVs and write DataPoint nodes to the graph.

    Returns the total number of nodes created.
    """
    suppliers = _build_suppliers(_read("suppliers.csv"))
    sites = _build_sites(_read("sites.csv"))
    skus = _build_skus(_read("skus.csv"), suppliers)
    pos = _build_purchase_orders(_read("purchase_orders.csv"), suppliers, skus, sites)
    shipments = _build_shipments(_read("shipments.csv"), pos, sites)
    tracking_events = _build_tracking_events(_read("tracking_events.csv"), shipments)
    carrier_lanes = _build_carrier_lanes(_read("carrier_performance.csv"))

    customer_orders = _build_customer_orders(_read("customer_orders.csv"), skus, sites, shipments)

    decision_rows = _read("planner_decisions.csv")
    decisions = _build_decisions(decision_rows, shipments, pos)
    outcomes = _build_outcomes(decision_rows, decisions)
    feedback = _build_feedback(_read("feedback_corrections.csv"), decisions, shipments)

    all_nodes = (
        list(suppliers.values())
        + list(sites.values())
        + list(skus.values())
        + list(pos.values())
        + list(shipments.values())
        + tracking_events
        + carrier_lanes
        + list(customer_orders.values())
        + list(decisions.values())
        + outcomes
        + feedback
    )

    await add_data_points(all_nodes)
    return len(all_nodes)
