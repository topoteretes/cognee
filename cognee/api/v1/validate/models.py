from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    type: str
    count: int = 0
    detail: str = ""


class ValidationReport(BaseModel):
    status: ValidationStatus = ValidationStatus.HEALTHY
    dataset: str = ""
    issues: List[ValidationIssue] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    checks_run: List[str] = Field(default_factory=list)

    def add_issue(
        self,
        severity: IssueSeverity,
        issue_type: str,
        count: int = 0,
        detail: str = "",
    ):
        self.issues.append(
            ValidationIssue(severity=severity, type=issue_type, count=count, detail=detail)
        )
        if severity == IssueSeverity.ERROR:
            self.status = ValidationStatus.UNHEALTHY
        elif severity == IssueSeverity.WARNING and self.status == ValidationStatus.HEALTHY:
            self.status = ValidationStatus.DEGRADED
