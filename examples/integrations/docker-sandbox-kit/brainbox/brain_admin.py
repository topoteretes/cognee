"""Host-side admin for brainbox: mint, scope, and revoke sandbox identities on the brain.

Talks to the brain's REST API with the OWNER's key (never given to a sandbox).
Every write here is reversible by `revoke`/`purge`.

    python brain_admin.py mint  --name brainbox-x --read <dataset name|uuid>...   -> JSON
    python brain_admin.py revoke --agent <uuid>       # ACL + policy rows removed, identity kept
    python brain_admin.py purge  --agent <uuid>       # revoke + delete the agent's datasets + agent
    python brain_admin.py datasets                    # list owner datasets (name, id, nodes)

The brain applies two layers to an agent read:
  1. the ACL grant (POST /api/v1/permissions/datasets/{agent}?permission_name=read).
     This layer exists on every cognee API server; agents reach datasets only
     through explicit grants.
  2. the owner's sharing policy (PUT /api/v1/workspace/learning/settings/sharing):
     per-agent allow/deny rules per dataset. `share_sessions: true` shares every
     session-bearing dataset with any agent that has no explicit rule, so `mint`
     writes an explicit rule for EVERY family dataset: allow for the granted
     ones, deny for the rest. Only the memory-workspace brain build (the server
     the cognee-memory plugin runs on :8011) has this route; a stock `dev`
     server answers 404 and this tool then works with layer 1 alone and says so
     (`"policy": "unavailable"` in its output).

What "revoke" can and cannot do: there is no owner-side route to invalidate
another principal's API key, so after `revoke` the agent still authenticates —
it just holds no grant and an explicit deny on every family dataset. `purge`
deletes the agent, which is the only call that kills the key. Grants are
removed before the agent is deleted in both paths (ACL rows do not cascade).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from uuid import UUID

DEFAULT_URL = os.environ.get("BRAIN_URL", "http://127.0.0.1:8011")
DEFAULT_KEY_FILE = os.environ.get(
    "BRAIN_OWNER_KEY_FILE", str(Path.home() / ".cognee-plugin" / "api_key.json")
)


def owner_key() -> str:
    key = os.environ.get("BRAIN_OWNER_KEY")
    if key:
        return key
    with open(DEFAULT_KEY_FILE) as handle:
        return json.load(handle)["api_key"]


class BrainError(SystemExit):
    """A non-2xx answer from the brain; `status` lets callers skip only what is expected."""

    def __init__(self, method: str, path: str, status: int, detail: str):
        super().__init__(f"{method} {path} -> HTTP {status}: {detail[:300]}")
        self.status = status


class Brain:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key
        self._policy_available: bool | None = None

    def call(self, method: str, path: str, body=None, timeout: float = 60.0):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            method=method,
            data=data,
            headers={"X-Api-Key": self.key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise BrainError(method, path, error.code, detail) from error

    # -- datasets -------------------------------------------------------------
    def datasets(self) -> list[dict]:
        return self.call("GET", "/api/v1/datasets")

    def resolve_dataset(self, ref: str) -> str:
        try:
            return str(UUID(ref))
        except ValueError:
            pass
        matches = [d["id"] for d in self.datasets() if d["name"] == ref]
        if not matches:
            raise SystemExit(f"dataset {ref!r} not found among the owner's datasets")
        return matches[0]

    # -- sharing policy -------------------------------------------------------
    def settings(self) -> dict | None:
        """The owner's workspace settings, or None when the brain has no sharing policy.

        The route only exists on the memory-workspace brain build; a stock cognee
        server answers 404. The outcome is reported once on stderr so a run against
        the wrong server does not fail on its first policy call without a hint.
        """
        if self._policy_available is False:
            return None
        try:
            state = self.call("GET", "/api/v1/workspace/learning/settings")
        except BrainError as error:
            if error.status != 404:
                raise
            self._policy_available = False
            print(
                f"note: {self.url} has no sharing-policy route "
                "(/api/v1/workspace/learning/settings -> 404). This is a stock cognee server; "
                "the per-agent allow/deny layer needs the memory-workspace brain build. "
                "Continuing with ACL grants only.",
                file=sys.stderr,
            )
            return None
        self._policy_available = True
        return state

    def family_dataset_ids(self) -> list[str]:
        """Every dataset an agent might hold a grant on: the policy's family, or all
        the owner's datasets when the brain has no policy layer."""
        state = self.settings()
        if state is not None:
            return [d["id"] for d in state["family_datasets"]]
        return [d["id"] for d in self.datasets()]

    def set_agent_rules(self, agent_id: str, rules: dict[str, str] | None) -> int | None:
        state = self.settings()
        if state is None:
            return None
        sharing = state["sharing"]
        value = sharing["value"]
        value.setdefault("agent_dataset_rules", {})
        if rules is None:
            value["agent_dataset_rules"].pop(agent_id, None)
        else:
            value["agent_dataset_rules"][agent_id] = rules
        saved = self.call(
            "PUT",
            "/api/v1/workspace/learning/settings/sharing",
            {"value": value, "revision": sharing["revision"]},
        )
        return saved["revision"]

    # -- agents ---------------------------------------------------------------
    def mint(self, name: str, read_ids: list[str]) -> dict:
        # Check the policy layer before creating anything, so a brain without it
        # is reported up front rather than after the identity exists.
        family = set(self.family_dataset_ids())
        agent = self.call("POST", f"/api/v1/agents/create?name={name}")
        agent_id = agent["agentId"]
        for dataset_id in read_ids:
            self.call(
                "POST",
                f"/api/v1/permissions/datasets/{agent_id}?permission_name=read",
                [dataset_id],
            )
        rules = {d: "deny" for d in family}
        for dataset_id in read_ids:
            rules[dataset_id] = "allow"
        revision = self.set_agent_rules(agent_id, rules)
        return {
            "agent_id": agent_id,
            "agent_email": agent["agentEmail"],
            "api_key": agent["agentApiKey"],
            "read": read_ids,
            "policy": "applied" if revision is not None else "unavailable",
            "policy_revision": revision,
            "denied": len(family - set(read_ids)),
        }

    def agent_datasets(self, agent_id: str) -> list[dict]:
        return [d for d in self.datasets() if d.get("ownerId") == agent_id]

    def drop_grants(self, agent_id: str, dataset_ids: list[str]) -> int:
        """Remove the agent's read and write grants on the given datasets.

        Removing a grant that does not exist succeeds on the brain, so the only
        answers worth skipping are 403/404 (a dataset the owner cannot administer,
        or one that is gone). Anything else — 401, 5xx — is a failed revoke and
        must surface, not be reported as success.
        """
        dropped = 0
        for permission in ("read", "write"):
            for dataset_id in dataset_ids:
                try:
                    self.call(
                        "DELETE",
                        f"/api/v1/permissions/datasets/{agent_id}?permission_name={permission}",
                        [dataset_id],
                    )
                    dropped += 1
                except BrainError as error:
                    if error.status not in (403, 404):
                        raise
        return dropped

    def revoke(self, agent_id: str) -> dict:
        family = self.family_dataset_ids()
        dropped = self.drop_grants(agent_id, family)
        revision = self.set_agent_rules(agent_id, {d: "deny" for d in family})
        return {
            "agent_id": agent_id,
            "grants_dropped": dropped,
            "policy": "applied" if revision is not None else "unavailable",
            "policy_revision": revision,
            # No owner-side route invalidates another principal's key: the
            # identity still authenticates, with no grant and an explicit deny
            # on every family dataset. `purge` is what kills the key.
            "kept_identity": True,
            "key_still_valid": True,
        }

    def purge(self, agent_id: str) -> dict:
        # Grants first: ACL rows do not cascade from the principal, and an agent
        # deleted with grants in place leaves orphaned rows behind.
        self.drop_grants(agent_id, self.family_dataset_ids())
        owned = self.agent_datasets(agent_id)
        for dataset in owned:
            self.call("DELETE", f"/api/v1/datasets/{dataset['id']}")
        self.set_agent_rules(agent_id, None)
        self.call("DELETE", f"/api/v1/agents/{agent_id}")
        return {
            "agent_id": agent_id,
            "deleted_datasets": [d["name"] for d in owned],
            "key_still_valid": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mint")
    p.add_argument("--name", required=True)
    p.add_argument("--read", nargs="*", default=[], help="dataset names or ids to grant read")
    p = sub.add_parser("revoke")
    p.add_argument("--agent", required=True)
    p = sub.add_parser("purge")
    p.add_argument("--agent", required=True)
    sub.add_parser("datasets")
    p = sub.add_parser("resolve")
    p.add_argument("ref")
    args = parser.parse_args()

    brain = Brain(args.url, owner_key())
    if args.cmd == "mint":
        out = brain.mint(args.name, [brain.resolve_dataset(r) for r in args.read])
    elif args.cmd == "revoke":
        out = brain.revoke(args.agent)
    elif args.cmd == "purge":
        out = brain.purge(args.agent)
    elif args.cmd == "resolve":
        out = {"id": brain.resolve_dataset(args.ref)}
    else:
        out = [
            {"name": d["name"], "id": d["id"], "owner": d.get("ownerId")} for d in brain.datasets()
        ]
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
