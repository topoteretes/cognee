"""In-memory stand-in for the BambooHR REST API, shaped like its real responses.

An endpoint the connector does not use is not answered: a call to it raises
``AssertionError`` and fails the test that made it.
"""

import re

from cognee_connector_bamboohr.source import (
    BambooHRAPIError,
    BambooHRNotFoundError,
    BambooHRRateLimitedError,
)

# The field list GET /meta/fields answers: every field carries a stable ``alias`` the
# report and the reads ask for, and a display ``name`` the rendering labels lines with.
META_FIELDS = [
    {"id": 1, "name": "First Name", "type": "text", "alias": "firstName"},
    {"id": 2, "name": "Last Name", "type": "text", "alias": "lastName"},
    {"id": 3, "name": "Job Title", "type": "text", "alias": "jobTitle"},
    {"id": 4, "name": "Department", "type": "list", "alias": "department"},
    {"id": 5, "name": "Location", "type": "text", "alias": "location"},
    {"id": 6, "name": "Status", "type": "list", "alias": "status"},
    {"id": 7, "name": "T-shirt size", "type": "text", "alias": "customTShirtSize"},
]
_EMPLOYEE_PATH = re.compile(r"/employees/([^/]+)/")


class FakeBambooHR:
    """One company: a field list, employees, and per-path failures to raise."""

    def __init__(self, employees: dict[str, dict] | None = None):
        self.meta_fields = [dict(field) for field in META_FIELDS]
        self.employees: dict[str, dict] = {
            str(employee_id): dict(record) for employee_id, record in (employees or {}).items()
        }
        # Readable but absent from the report, like an employee a report setup skips.
        self.hidden: set[str] = set()
        # Path -> exception raised for it, like a 403 on the report or a 500 on a read.
        self.failures: dict[str, Exception] = {}
        self.rate_limited = False
        self.calls: list[tuple[str, str]] = []
        # Everything asked, so a test can assert the allowlist is what reaches the wire.
        self.request_log: list[dict] = []

    # -- seeding -----------------------------------------------------------
    def add_employee(self, employee_id: str | int, **fields) -> None:
        self.employees[str(employee_id)] = dict(fields)

    def remove_employee(self, employee_id: str | int) -> None:
        """The employee leaves the report; a read answers what the failure map says."""
        self.employees.pop(str(employee_id), None)

    # -- the API ------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        body: dict | None = None,
        check_active=None,
    ):
        if check_active is not None:
            check_active()
        self.calls.append((method, path))
        self.request_log.append({"method": method, "path": path, "params": params, "body": body})
        if self.rate_limited:
            raise BambooHRRateLimitedError(path)
        failure = self.failures.get(path)
        if failure is not None:
            raise failure
        if path == "/meta/fields":
            return [dict(field) for field in self.meta_fields]
        if path == "/employees/custom-report":
            return self._report(body)
        match = _EMPLOYEE_PATH.fullmatch(path)
        if match:
            return self._employee(match.group(1), params)
        raise AssertionError(f"FakeBambooHR got a request it does not model: {method} {path}")

    def _report(self, body: dict | None) -> list[dict]:
        assert isinstance(body, dict) and isinstance(body.get("fields"), list)
        rows = []
        for employee_id in sorted(self.employees, key=int):
            if employee_id in self.hidden:
                continue
            record = self.employees[employee_id]
            rows.append(
                {
                    "employeeId": employee_id,
                    **{field: record.get(field) for field in body["fields"]},
                }
            )
        return rows

    def _employee(self, employee_id: str, params: dict | None) -> dict:
        if employee_id not in self.employees:
            raise BambooHRNotFoundError(f"/employees/{employee_id}/")
        record = self.employees[employee_id]
        requested = (params or {}).get("fields", "")
        return {field: record[field] for field in requested.split(",") if field in record}
