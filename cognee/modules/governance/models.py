from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict

class ODRLPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    uid: str
    type: Literal["Set", "Offer", "Agreement"] = "Set"
    assigner: str       # tenant id as URI: "urn:cognee:tenant:{id}"
    assignee: str       # principal id as URI: "urn:cognee:user:{id}"
    target: str         # dataset id as URI: "urn:cognee:dataset:{id}"
    action: str         # ODRL action URI from the mapping table in the RFC
    custom_action: Optional[str] = None  # for non-ODRL cognee actions

from uuid import UUID
from datetime import datetime

class AuditRecord(BaseModel):
    actor_id: UUID | str
    action: str
    target_dataset_id: UUID | str
    timestamp: datetime | str
    outcome: Literal["ALLOWED", "DENIED"]
    policy_id: Optional[UUID | str] = None
    denial_reason: Optional[str] = None
    previous_hash: Optional[str] = None
    row_hash: str

class GovernanceBundle(BaseModel):
    schema_version: str = "1.0"
    odrl_context: str = "https://www.w3.org/ns/odrl/2/"
    schema_context: str = "https://schema.org/"
    exported_at: str
    dataset_id: str
    permission_model: list[ODRLPolicy]
    decision_history: list[AuditRecord]
    rejection_trail: list[AuditRecord]
    bundle_hash: str
