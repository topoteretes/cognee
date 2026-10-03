"""Host-side admin for brainbox: mint, scope, and revoke sandbox identities on the brain.

Talks to the brain's REST API with the OWNER's key (never given to a sandbox).
Every write here is reversible by `revoke`/`purge`.

    python brain_admin.py mint  --name brainbox-x --read <dataset name|uuid>...   -> JSON
    python brain_admin.py revoke --agent <uuid>       # ACL + policy rows removed, identity kept
    python brain_admin.py purge  --agent <uuid>       # revoke + delete the agent's datasets + agent
    python brain_admin.py datasets                    # list owner datasets (name, id, nodes)

The brain applies two layers to an agent read:
  1. the ACL grant (POST /api/v1/permissions/datasets/{agent}?permission_name=read)
  2. the owner's sharing policy (PUT /api/v1/workspace/learning/settings/sharing):
     per-agent allow/deny rules per dataset. `share_sessions: true` shares every
     session-bearing dataset with any agent that has no explicit rule, so `mint`
     writes an explicit rule for EVERY family dataset: allow for the granted
     ones, deny for the rest.
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


class Brain:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key

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
            raise SystemExit(f"{method} {path} -> HTTP {error.code}: {detail[:300]}") from error

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
    def settings(self) -> dict:
        return self.call("GET", "/api/v1/workspace/learning/settings")

    def set_agent_rules(self, agent_id: str, rules: dict[str, str] | None) -> int:
        state = self.settings()
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
        agent = self.call("POST", f"/api/v1/agents/create?name={name}")
        agent_id = agent["agentId"]
        for dataset_id in read_ids:
            self.call(
                "POST",
                f"/api/v1/permissions/datasets/{agent_id}?permission_name=read",
                [dataset_id],
            )
        family = {d["id"] for d in self.settings()["family_datasets"]}
        rules = {d: "deny" for d in family}
        for dataset_id in read_ids:
            rules[dataset_id] = "allow"
        revision = self.set_agent_rules(agent_id, rules)
        return {
            "agent_id": agent_id,
            "agent_email": agent["agentEmail"],
            "api_key": agent["agentApiKey"],
            "read": read_ids,
            "policy_revision": revision,
            "denied": len(family) - len(read_ids),
        }

    def agent_datasets(self, agent_id: str) -> list[dict]:
        return [d for d in self.datasets() if d.get("ownerId") == agent_id]

    def revoke(self, agent_id: str) -> dict:
        # Every dataset the agent might hold a grant on: revoke read and write.
        family = [d["id"] for d in self.settings()["family_datasets"]]
        for permission in ("read", "write"):
            for dataset_id in family:
                try:
                    self.call(
                        "DELETE",
                        f"/api/v1/permissions/datasets/{agent_id}?permission_name={permission}",
                        [dataset_id],
                    )
                except SystemExit:
                    pass  # no such grant
        rules = {d: "deny" for d in family}
        revision = self.set_agent_rules(agent_id, rules)
        return {"agent_id": agent_id, "policy_revision": revision, "kept_identity": True}

    def purge(self, agent_id: str) -> dict:
        owned = self.agent_datasets(agent_id)
        for dataset in owned:
            self.call("DELETE", f"/api/v1/datasets/{dataset['id']}")
        self.set_agent_rules(agent_id, None)
        self.call("DELETE", f"/api/v1/agents/{agent_id}")
        return {"agent_id": agent_id, "deleted_datasets": [d["name"] for d in owned]}


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
