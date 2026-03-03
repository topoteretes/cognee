"""Supply chain domain models built on Cognee's DataPoint base class.

Each model represents a node type in the knowledge graph. Relationships between
models become typed edges that Cognee can traverse for multi-hop queries.

The ``metadata["index_fields"]`` list controls which string fields are embedded
for vector search — choose the fields most useful for semantic retrieval.

IMPORTANT: Relationship fields use concrete type hints (e.g. ``list[Supplier]``)
rather than ``SkipValidation[Any]`` so that the LLM can generate properly-typed
nested objects when ``cognify(graph_model=SupplyChainContext)`` is used.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import model_validator

from cognee.infrastructure.engine import DataPoint

_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class SupplyChainDataPoint(DataPoint):
    """Demo-only base class with LLM-output tolerance.

    When ``cognify(graph_model=SupplyChainContext)`` asks the LLM to
    produce typed entities, two common failure modes occur:

    1. The LLM puts a domain ID (e.g. ``"SHP-003"``) in the ``id``
       field instead of a UUID.  We convert it to a deterministic
       UUID via ``uuid5`` so the same entity always maps to the same
       node — and matches nodes already built by ``csv_to_graph.py``.

    2. The LLM returns ``null`` for non-optional ``str`` fields it
       cannot fill.  We coerce those to ``""`` so validation passes.

    Both fixes run in ``mode="before"`` (on the raw dict) so they
    prevent instructor retry loops and cost zero extra LLM calls.
    """

    @model_validator(mode="before")
    @classmethod
    def _sanitize_llm_output(cls, data):
        if not isinstance(data, dict):
            return data

        raw_id = data.get("id")
        if isinstance(raw_id, str):
            try:
                uuid.UUID(raw_id)
            except ValueError:
                data["id"] = uuid.uuid5(_NS, f"{cls.__name__}:{raw_id}")

        for field_name, field_info in cls.model_fields.items():
            if field_name == "id":
                continue
            if data.get(field_name) is None and field_info.annotation is str:
                data[field_name] = ""

        return data

    @model_validator(mode="after")
    def _ensure_index_fields_non_empty(self):
        for field_name in self.metadata.get("index_fields", []):
            val = getattr(self, field_name, None)
            if isinstance(val, str) and not val.strip():
                object.__setattr__(self, field_name, "n/a")
        return self


class Supplier(SupplyChainDataPoint):
    supplier_id: str
    name: str
    country: str
    tier: str = ""
    certifications: str = ""
    reliability_score: float = 0.0
    avg_lead_time_days: int = 0
    payment_terms: str = ""
    preferred_freight: str = ""
    notes: str = ""

    metadata: dict = {"index_fields": ["name", "notes"]}


class Site(SupplyChainDataPoint):
    site_id: str
    name: str
    region: str = ""
    city: str = ""

    metadata: dict = {"index_fields": ["name", "city"]}


class SKU(SupplyChainDataPoint):
    sku_id: str
    description: str
    category: str = ""
    criticality: str = ""
    unit_cost: float = 0.0
    sourcing_type: str = ""
    special_handling: str = ""
    safety_stock: str = ""

    components: Optional[List[SKU]] = None
    supplied_by: Optional[List[Supplier]] = None

    metadata: dict = {"index_fields": ["description", "special_handling", "safety_stock"]}


SKU.model_rebuild()


class PurchaseOrder(SupplyChainDataPoint):
    po_number: str
    status: str = ""
    quantity: int = 0
    incoterm: str = ""
    due_date: str = ""

    sku: Optional[SKU] = None
    supplier: Optional[Supplier] = None
    destination: Optional[Site] = None

    metadata: dict = {"index_fields": ["po_number", "status"]}


class Shipment(SupplyChainDataPoint):
    shipment_id: str
    carrier: str = ""
    status: str = ""
    origin_port: str = ""
    destination_port: str = ""
    freight_mode: str = ""
    expected_delivery: str = ""
    actual_delivery: str = ""
    total_delay_days: int = 0
    notes: str = ""

    purchase_order: Optional[PurchaseOrder] = None
    origin_site: Optional[Site] = None
    destination_site: Optional[Site] = None

    metadata: dict = {"index_fields": ["shipment_id", "notes"]}


class TrackingEvent(SupplyChainDataPoint):
    event_type: str
    timestamp: str
    location: str = ""
    port_code: str = ""
    delay_minutes: int = 0
    delay_reason: str = ""
    details: str = ""

    shipment: Optional[Shipment] = None

    metadata: dict = {"index_fields": ["details"]}


class CarrierLane(SupplyChainDataPoint):
    carrier_name: str
    lane: str = ""
    period: str = ""
    total_shipments: int = 0
    otd_percent: float = 0.0
    avg_transit_days: float = 0.0
    congestion_events: int = 0
    demurrage_claims: int = 0
    rate_info: str = ""
    notes: str = ""

    metadata: dict = {"index_fields": ["carrier_name", "lane", "notes"]}


class CustomerOrder(SupplyChainDataPoint):
    order_id: str
    customer_name: str
    priority: str = ""
    order_value: float = 0.0
    delivery_deadline: str = ""
    penalty_per_day: float = 0.0
    penalty_percent_per_day: float = 0.0
    penalty_cap_days: int = 0
    penalty_details: str = ""
    relationship_notes: str = ""

    required_skus: Optional[List[SKU]] = None
    destination_site: Optional[Site] = None
    fulfilled_by: Optional[List[Shipment]] = None

    metadata: dict = {"index_fields": ["customer_name", "penalty_details", "relationship_notes"]}


class Decision(SupplyChainDataPoint):
    decision_date: str = ""
    action_taken: str
    rationale: str = ""
    cost_impact: str = ""
    approved_by: str = ""

    applies_to: Optional[Shipment] = None
    applies_to_po: Optional[PurchaseOrder] = None

    metadata: dict = {"index_fields": ["action_taken", "rationale"]}


class Outcome(SupplyChainDataPoint):
    outcome_date: str = ""
    metric: str = ""
    result_description: str
    value_before: str = ""
    value_after: str = ""

    decision: Optional[Decision] = None

    metadata: dict = {"index_fields": ["result_description"]}


class Feedback(SupplyChainDataPoint):
    feedback_date: str = ""
    rating: int = 0
    original_answer: str = ""
    correction: str
    source_context: str = ""

    corrects: Optional[Decision] = None
    related_shipment: Optional[Shipment] = None

    metadata: dict = {"index_fields": ["correction", "original_answer"]}


class LessonLearned(SupplyChainDataPoint):
    lesson_date: str = ""
    description: str
    recommendation: str = ""
    historical_reference: str = ""

    related_suppliers: Optional[List[Supplier]] = None
    related_skus: Optional[List[SKU]] = None

    metadata: dict = {"index_fields": ["description", "recommendation"]}


class SupplyChainContext(SupplyChainDataPoint):
    """Root model passed to ``cognify(graph_model=...)`` so the LLM extracts
    typed, relationship-rich entities instead of the generic KnowledgeGraph.

    Each chunk may populate only a subset of the lists below — that is fine.
    The graph engine recursively creates nodes and edges from every nested
    DataPoint, so cross-references (e.g. CustomerOrder.fulfilled_by → Shipment)
    become first-class traversable edges.
    """

    summary: str = ""

    suppliers: List[Supplier] = []
    skus: List[SKU] = []
    sites: List[Site] = []
    purchase_orders: List[PurchaseOrder] = []
    shipments: List[Shipment] = []
    tracking_events: List[TrackingEvent] = []
    carrier_lanes: List[CarrierLane] = []
    customer_orders: List[CustomerOrder] = []
    decisions: List[Decision] = []
    outcomes: List[Outcome] = []
    feedbacks: List[Feedback] = []
    lessons_learned: List[LessonLearned] = []

    metadata: dict = {"index_fields": ["summary"]}


SupplyChainContext.model_rebuild()
