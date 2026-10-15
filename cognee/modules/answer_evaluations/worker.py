"""Run via cron (local or cloud); one shared service, no scheduler in web workers."""

import argparse
import asyncio
import os
import random
import secrets
import statistics
import time
import uuid

from sqlalchemy import select

from .models import EvaluationRun, EvaluationScope, EvaluationTrace
from .privacy import Protection
from .schema import score
from .service import lock_scope, prune
from .warehouse import Warehouse


async def process_scope(sessions, warehouse, protection, scope_id, judge, *, now=None):
    now = time.time() if now is None else now
    async with sessions() as session:
        scope = await lock_scope(session, scope_id, "", "")
        if scope is None:
            return
        await prune(session, scope, now)
        old_runs = (
            await session.scalars(
                select(EvaluationRun).where(
                    EvaluationRun.scope_id == scope_id,
                    EvaluationRun.status == "running",
                    EvaluationRun.created_at < now - 3600,
                )
            )
        ).all()
        for old_run in old_runs:
            old_run.status = "abandoned"

        rows = list(
            (
                await session.scalars(
                    select(EvaluationTrace).where(
                        EvaluationTrace.scope_id == scope_id,
                        EvaluationTrace.ciphertext.is_not(None),
                    )
                )
            ).all()
        )
        # Serialize reconciliation with admission/deletion across API processes.
        await asyncio.to_thread(warehouse.replace, scope_id, rows, now)
        scope.synced_revision = scope.revision
        if not scope.enabled or scope.last_run_at > now - 86400 or judge is None:
            await session.commit()
            return
        eligible = [r for r in rows if protection.decrypt(r.ciphertext).get("evidence")]
        if len(eligible) < 20:
            await session.commit()
            return
        seed = secrets.randbits(63)
        chosen = random.Random(seed).sample(eligible, 20)
        selected = [{"trace_id": r.id, "version": r.fingerprint} for r in chosen]
        run_id = str(uuid.uuid4())
        session.add(
            EvaluationRun(
                id=run_id,
                scope_id=scope_id,
                created_at=now,
                status="running",
                result={
                    "seed": seed,
                    "selected": selected,
                    "verdicts": [],
                    "rubric_version": "sampled-evidence-v1",
                    "recall_budget_seconds": 45,
                },
            )
        )
        scope.last_run_at = now  # A crashed run is not automatically replayed/charged twice.
        await session.commit()

    for index, item in enumerate(selected):
        async with sessions() as session:
            scope = await session.get(EvaluationScope, scope_id)
            run = await session.get(EvaluationRun, run_id)
            trace = await session.get(EvaluationTrace, item["trace_id"])
            if run is None or not scope.enabled:
                return  # Revocation/deletion wins over an in-flight evaluation.
            cipher = await asyncio.to_thread(warehouse.get, scope_id, item["trace_id"], time.time())
            if (
                trace is None
                or not trace.ciphertext
                or not cipher
                or trace.expires_at <= time.time()
            ):
                run.status = "invalidated"
                await session.commit()
                return
            payload = protection.decrypt(cipher)
        try:
            verdict = await asyncio.wait_for(judge(payload, bool((seed + index) % 2)), timeout=150)
            if verdict.get("verdict") not in {"better", "equivalent", "worse", "unjudgeable"}:
                raise ValueError("Invalid evaluation verdict")
            # Allowlist only non-content metrics. Judge output cannot persist arbitrary text.
            safe = {
                k: v
                for k, v in verdict.items()
                if k
                in {
                    "verdict",
                    "replay_tokens",
                    "judge_tokens",
                    "cost_usd",
                    "cost_complete",
                    "baseline_latency_ms",
                    "memory_latency_ms",
                    "method",
                    "model",
                    "judge_model",
                }
            }
        except (ValueError, KeyError, TypeError, OSError, TimeoutError):
            async with sessions() as session:
                run = await session.get(EvaluationRun, run_id)
                if run:
                    run.status = "failed"
                    await session.commit()
            return  # No response bodies or prompts in logs; no automatic paid retries.
        async with sessions() as session:
            scope = await lock_scope(session, scope_id, "", "")
            run = await session.get(EvaluationRun, run_id)
            if run is None or not scope.enabled:
                return
            data = {**run.result, "verdicts": run.result["verdicts"] + [{**item, **safe}]}
            run.result = data
            await session.commit()

    async with sessions() as session:
        await lock_scope(session, scope_id, "", "")
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            return
        counts = {
            k: sum(v["verdict"] == k for v in run.result["verdicts"])
            for k in ("better", "equivalent", "worse", "unjudgeable")
        }
        summary = {
            "pairs": 20,
            "cases": 20,
            **counts,
            "score": score(counts["better"], counts["worse"], 20),
            "improvement_rate": counts["better"] / 20,
            "regression_rate": counts["worse"] / 20,
            "recall_budget_seconds": 45,
        }
        verdicts = run.result["verdicts"]
        for arm in ("baseline", "memory"):
            values = sorted(v[arm + "_latency_ms"] for v in verdicts if arm + "_latency_ms" in v)
            if len(values) == 20:
                summary[arm + "_p50_latency_ms"] = statistics.median(values)
                summary[arm + "_p95_latency_ms"] = values[18]
        summary["total_tokens"] = sum(
            v.get("replay_tokens", 0) + v.get("judge_tokens", 0) for v in verdicts
        )
        summary["cost_complete"] = all(v.get("cost_complete") is True for v in verdicts)
        summary["cost_usd"] = (
            sum(v.get("cost_usd", 0) for v in verdicts) if summary["cost_complete"] else None
        )
        summary["cost_basis"] = "configured_rates"
        run.result = {**run.result, "summary": summary}
        run.status = "complete" if not counts["unjudgeable"] else "insufficient_evidence"
        await session.commit()
    return run_id


async def sweep(*, evaluate=False):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.permissions.methods.get_permitted_dataset_ids import (
        get_permitted_dataset_ids,
    )

    from .judge import MatchedJudge

    sessions = get_relational_engine().get_async_session
    protection, warehouse = Protection(), Warehouse.configured()
    judge = None
    configuration_error = None
    if evaluate:
        try:
            judge = MatchedJudge()
        except (KeyError, ValueError) as exc:
            configuration_error = exc
    try:
        async with sessions() as session:
            scopes = list((await session.scalars(select(EvaluationScope))).all())
        for scope in scopes:
            # Dataset/user removal and revoked read permission purge traces as well.
            allowed = await get_permitted_dataset_ids(uuid.UUID(scope.owner_id))
            if uuid.UUID(scope.dataset_id) not in allowed:
                from .service import erase

                async with sessions() as session:
                    row = await lock_scope(session, scope.id, "", "")
                    row.enabled = False
                    await erase(session, row)
                    await session.commit()
            await process_scope(sessions, warehouse, protection, scope.id, judge)
        if configuration_error:
            raise ValueError(
                "Evaluation provider configuration missing/invalid; storage reconciliation completed"
            ) from None
    finally:
        warehouse.connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Allow daily paid evaluation")
    parser.add_argument(
        "--loop", action="store_true", help="Reconcile every 60s; evaluate at most daily"
    )
    args = parser.parse_args()
    if args.evaluate and os.environ.get("COGNEE_EVALUATIONS_ENABLED", "false").lower() != "true":
        parser.error("Set COGNEE_EVALUATIONS_ENABLED=true before allowing paid evaluation")
    while True:
        asyncio.run(sweep(evaluate=args.evaluate))
        if not args.loop:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
