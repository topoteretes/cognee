"""RL rollouts on Daytona with cognee memory as the thing that persists across episodes.

The shape is the HUD cookbook's (``hud-rl-cookbook``): a task template with
variants, one fresh sandbox per attempt, a pytest grader with a binary reward,
groups of attempts per variant. What this adds is memory. The cookbook's
sandboxes are ephemeral and every attempt starts from the same seeded state;
here one long-lived **memory sandbox** runs the cognee API server, and the
agent loop on the host recalls lessons from it before an attempt and writes
the attempt back, scored, afterwards. ``improve()`` after each step distills
the scored attempts into lessons the next step recalls::

    memory sandbox   cognee-rl-memory-<run>   cognee API server, LLM key as a Secret, lives all run
    rollout sandbox  cognee-rl-<variant>-<n>  1 vCPU / 1 GB, seeded bug + hidden tests, deleted after grading

    for step in steps:                       # training steps
        for variant in step:                 # one per bug family
            lessons = memory.recall(family)  # once per group
            group   = N attempts, in parallel sandboxes: prompt (+lessons) -> LLM -> write -> pytest
            memory.remember(each attempt, reward)            # scored QA entry in a session
        memory.improve(session_ids=[step])                    # lessons for the next step
    evaluate held-out variants with memory off and on         # what memory alone buys
    export the memory as a COGX archive                       # the checkpoint

No gradient step runs here: that is the cookbook's hosted trainer, fed by
graded rollouts; the numbers this prints — pass rate with memory off and on,
and how many groups have mixed rewards (the ones a group-relative trainer can
learn from) — are what memory changes for that trainer. Partial credit is
deliberately not used (the cookbook found it flattens the within-group spread).

Prerequisites::

    pip install daytona httpx litellm
    export DAYTONA_API_KEY=...
    export LLM_API_KEY=sk-...        # the host-side policy calls; also the Secret on first run

Usage::

    python rl_loop.py                        # 3 steps x 4 variants x 8 attempts, eval, export
    python rl_loop.py --steps 1 --group 4    # a quick one
    python rl_loop.py --keep                 # keep the memory sandbox for inspection
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import httpx
import litellm
from daytona import (  # type: ignore[import-untyped]
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaError,
    Image,
    ListSandboxesQuery,
    Resources,
    Sandbox,
    SessionExecuteRequest,
)
from openai import OpenAIError  # litellm's errors subclass it; openai comes with litellm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # ledger_env
sys.path.insert(0, str(HERE.parent))  # handover: snapshot, Secret, sandbox posture, COGX helper

import ledger_env as env  # noqa: E402
from daytona_api_client import SnapshotState  # type: ignore[import-untyped]  # noqa: E402
from handover import (  # noqa: E402
    AUTO_STOP_MINUTES,
    HELPER,
    HELPER_NAME,
    LLM_MODEL,
    LLM_VAULT_ENTRY,
    SNAPSHOT_NAME,
    activate_snapshot,
    ensure_secret,
    ensure_snapshot,
    run_in,
)

ENV_SNAPSHOT = "cognee-rl-env"  # python + pytest, 1 vCPU / 1 GB / 3 GB: the cookbook's sandbox size
DATASET = "rl_memory"
KILL_PYTHON = (
    'for p in /proc/[0-9]*; do grep -q python "$p/comm" 2>/dev/null && kill -9 "${p#/proc/}"; '
    "done; true"
)
ATTEMPT_DEADLINE = 240  # seconds per attempt: create + seed + grade + LLM + grade + delete
UPLOAD_TIMEOUT = 60  # seconds per file upload (the SDK's default is 30 minutes)
MEMORY_PORT = 8000
POLICY_MODEL = LLM_MODEL  # the host-side policy; cognee inside the memory sandbox uses the same


class AttemptAborted(Exception):
    """An attempt stopped between steps: past its deadline, or its group gave up on it."""


def labels(run_id: str, role: str) -> dict[str, str]:
    return {"app": "cognee", "demo": "rl", "run": run_id, "role": role}


# --------------------------------------------------------------------------- memory sandbox


def ensure_env_snapshot(daytona: Daytona) -> None:
    """The rollout image: Debian, Python 3.12, pytest. Built once, 1 GB like the cookbook's."""
    try:
        snapshot = daytona.snapshot.get(ENV_SNAPSHOT)
    except DaytonaError:
        snapshot = None
    if snapshot is not None:
        if snapshot.state == SnapshotState.ACTIVE:
            print(f"=== snapshot '{ENV_SNAPSHOT}' exists, reusing ===")
        else:
            print(
                f"=== snapshot '{ENV_SNAPSHOT}' exists but is {snapshot.state.value}: activating ==="
            )
            activate_snapshot(daytona, name=ENV_SNAPSHOT)
        return
    print(f"=== building snapshot '{ENV_SNAPSHOT}' (python + pytest; about a minute) ===")
    daytona.snapshot.create(
        CreateSnapshotParams(
            name=ENV_SNAPSHOT,
            image=Image.debian_slim("3.12").pip_install("pytest"),
            resources=Resources(cpu=1, memory=1, disk=3),
        ),
        on_logs=lambda line: print(f"[snapshot] {line}", end="", flush=True),
    )
    print()


# How a recall is answered: lessons, not a replay of a stored attempt. The
# context holds past attempts with their verdicts; the answer must be what the
# hidden tests turned out to expect.
LESSONS_SYSTEM_PROMPT = (
    "You distill lessons from earlier attempts at a coding task. The context contains past "
    "attempts: what was changed, whether the hidden tests passed (reward 1) or failed "
    "(reward 0), and for failures the assertion lines (expected values, expected exception "
    "types and messages, DID NOT RAISE). Answer with short bullet points stating the "
    "conventions the hidden tests enforce and the fixes that passed. Quote exact expected "
    "values and error messages. Never return a code file. If the context has nothing about "
    "the question, answer exactly: NO_LESSONS."
)

_NON_ANSWER = (
    "no_lessons",
    "warming up",
    "no knowledge graph data",
    "no information",
    "not contain",
    "no relevant",
    "does not provide",
    "cannot determine",
)


def _is_non_answer(text: str) -> bool:
    """A completion over an empty or unrelated graph says so instead of giving lessons."""
    low = text.lower()
    return len(text) < 60 or any(phrase in low for phrase in _NON_ANSWER) or text.startswith("```")


class Memory:
    """The cognee API server in the memory sandbox, reached over its private preview URL."""

    def __init__(self, url: str, token: str):
        self.http = httpx.Client(
            base_url=url, headers={"X-Daytona-Preview-Token": token}, timeout=600
        )

    def healthy(self) -> bool:
        try:
            return self.http.get("/health", timeout=10).status_code == 200
        except httpx.HTTPError:
            return False

    def remember(self, entry: dict, session_id: str) -> None:
        r = self.http.post(
            "/api/v1/remember/entry",
            json={"entry": entry, "dataset_name": DATASET, "session_id": session_id},
        )
        r.raise_for_status()

    def recall(self, query: str) -> str:
        r = self.http.post(
            "/api/v1/recall",
            json={
                "query": query,
                "datasets": [DATASET],
                "top_k": 10,
                "system_prompt": LESSONS_SYSTEM_PROMPT,
            },
        )
        if r.status_code == 404:
            return ""  # no dataset yet on the first step
        # Anything else (a 500, a 422, a 502 from the preview proxy) must fail
        # loudly: read as "no lessons", it would make the memory-on evaluation
        # quietly run without memory.
        r.raise_for_status()
        items = r.json()
        text = "\n".join(str(i.get("text", "")) for i in items if isinstance(i, dict)).strip()
        return "" if _is_non_answer(text) else text

    def improve(self, session_ids: list[str]) -> list[dict]:
        r = self.http.post(
            "/api/v1/improve", json={"dataset_name": DATASET, "session_ids": session_ids}
        )
        r.raise_for_status()
        return r.json().get("stages", [])


def start_memory_sandbox(daytona: Daytona, run_id: str) -> tuple[Sandbox, Memory]:
    """One sandbox from the cognee snapshot running the API server for the whole run.

    Single-user mode inside (no cognee auth): the sandbox is private, so the
    only way in is the preview URL with its token. The LLM key is the same
    host-scoped Secret the handover demo uses; the sandbox sees a placeholder.
    """
    name = f"cognee-rl-memory-{run_id}"
    print(f"=== memory sandbox: {name} (cognee API server) ===")
    sandbox = daytona.create(
        CreateSandboxFromSnapshotParams(
            snapshot=SNAPSHOT_NAME,
            name=name,
            secrets={"LLM_API_KEY": LLM_VAULT_ENTRY},
            labels=labels(run_id, "memory"),
            auto_stop_interval=2 * AUTO_STOP_MINUTES,  # idle through a memory-off evaluation
        ),
        timeout=180,
    )
    home = sandbox.get_user_home_dir()
    run_in(sandbox, f"mkdir -p {home}/memory-state {home}/rl")
    sandbox.fs.upload_file(HELPER.read_bytes(), f"{home}/rl/{HELPER_NAME}")
    server_env = {
        "DATA_ROOT_DIRECTORY": f"{home}/memory-state/data",
        "SYSTEM_ROOT_DIRECTORY": f"{home}/memory-state/system",
        "LLM_MODEL": LLM_MODEL,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TELEMETRY_DISABLED": "1",
        "LOG_LEVEL": "ERROR",
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "REQUIRE_AUTHENTICATION": "false",
        # Recalls here are lesson lookups, not conversation turns: no per-turn
        # feedback analysis (one LLM call per recall otherwise).
        "AUTO_FEEDBACK": "false",
        "HTTP_API_HOST": "0.0.0.0",
        "HTTP_API_PORT": str(MEMORY_PORT),
    }
    exports = " ".join(f"{k}={v}" for k, v in server_env.items())
    sandbox.process.create_session("api")
    sandbox.process.execute_session_command(
        "api",
        SessionExecuteRequest(
            command=f"cd {home} && env {exports} python3 -m cognee.api.client > {home}/api.log 2>&1",
            run_async=True,
        ),
    )
    preview = sandbox.get_preview_link(MEMORY_PORT)
    memory = Memory(preview.url, preview.token)
    print(f"    api: {preview.url} (private; preview token on every request)")
    t0 = time.time()
    while not memory.healthy():
        if time.time() - t0 > 240:
            log = sandbox.process.exec(f"tail -n 30 {home}/api.log").result or ""
            raise RuntimeError(f"cognee API server did not come up:\n{log}")
        time.sleep(3)
    print(f"    healthy after {time.time() - t0:.0f}s; dataset '{DATASET}', single-user mode")
    return sandbox, memory


# --------------------------------------------------------------------------- rollouts


class Rollouts:
    """Fresh sandboxes from the env snapshot, one per attempt, capped in parallel."""

    def __init__(self, daytona: Daytona, run_id: str, parallel: int):
        self.daytona = daytona
        self.run_id = run_id
        self.parallel = parallel
        self.slots = threading.Semaphore(parallel)
        self.lock = threading.Lock()
        self.timings: list[float] = []

    def attempt(self, variant_id: str, index: int, lessons: str, cancel: threading.Event) -> dict:
        """One episode: seed, grade, prompt the policy, write its answer, grade again.

        The deadline starts once the attempt holds a parallel slot, so waiting
        for one is not counted against it. Between steps it checks the deadline
        and ``cancel`` (set when its group stops waiting) and raises
        AttemptAborted, which deletes its sandbox and frees the slot instead of
        running the remaining calls for nothing.
        """
        with self.slots:
            t0 = time.time()
            deadline = time.monotonic() + ATTEMPT_DEADLINE

            def check() -> None:
                if cancel.is_set():
                    raise AttemptAborted("its group stopped waiting")
                if time.monotonic() > deadline:
                    raise AttemptAborted(f"past its {ATTEMPT_DEADLINE}s deadline")

            check()  # the group may have given up while this attempt waited for a slot
            sandbox = self._create(f"cognee-rl-{variant_id.lower()}-{index}")
            try:
                check()
                home = sandbox.get_user_home_dir()
                ws = f"{home}/{env.WORKSPACE}"
                run_in_quiet(sandbox, f"mkdir -p {ws}")
                files = env.seed(variant_id)
                for name, text in files.items():
                    sandbox.fs.upload_file(text.encode(), f"{ws}/{name}", timeout=UPLOAD_TIMEOUT)
                check()
                before = grade(sandbox, ws)
                failing = env.failing_tests(before)
                prompt = env.task_prompt(variant_id, files[env.SOURCE_FILE], failing, lessons)
                check()
                answer = policy(prompt)
                source = env.extract_source(answer)
                if source is None:
                    after, reward = "no python block in the answer", 0
                else:
                    check()
                    sandbox.fs.upload_file(
                        source.encode(), f"{ws}/{env.SOURCE_FILE}", timeout=UPLOAD_TIMEOUT
                    )
                    after = grade(sandbox, ws)
                    reward = int(env.passed(after))
            finally:
                try:
                    self.daytona.delete(sandbox)
                except DaytonaError:
                    pass
            elapsed = time.time() - t0
            with self.lock:
                self.timings.append(elapsed)
        return {
            "variant": variant_id,
            "index": index,
            "failing_before": failing,
            "seeded": files[env.SOURCE_FILE],
            "prompt": env.task_prompt(variant_id, files[env.SOURCE_FILE], failing, ""),
            "answer": source or answer,
            "output": after,
            "reward": reward,
            "seconds": elapsed,
            "timed_out": False,
        }

    def group(self, variant_id: str, size: int, lessons: str) -> list[dict]:
        """``size`` attempts in parallel under one deadline for the group.

        A sandbox call or an LLM call can hang on a dropped connection; a group
        must not. The group waits at most one attempt deadline per wave of
        ``parallel`` attempts, then tells the stragglers to stop. Attempts that
        timed out are marked ``timed_out`` so the run reports them apart from
        the policy's failures; any rollout sandbox of this run still alive
        afterwards is deleted by label.
        """
        cancel = threading.Event()
        waves = -(-size // self.parallel)  # ceil: attempts run `parallel` at a time
        group_deadline = ATTEMPT_DEADLINE * waves
        pool = ThreadPoolExecutor(max_workers=size)
        futures = [pool.submit(self.attempt, variant_id, i, lessons, cancel) for i in range(size)]
        done, _ = wait(futures, timeout=group_deadline)
        cancel.set()
        results = []
        for i, future in enumerate(futures):
            if future not in done:
                print(
                    f"    attempt {i} still running after the group's {group_deadline}s; timed out"
                )
                results.append(self._timed_out(variant_id, i, float(group_deadline)))
                continue
            try:
                results.append(future.result())
            except AttemptAborted as err:
                print(f"    attempt {i} stopped: {err}; timed out")
                results.append(self._timed_out(variant_id, i, float(ATTEMPT_DEADLINE)))
            except (OpenAIError, DaytonaError, RuntimeError, httpx.HTTPError, OSError) as err:
                # One attempt's LLM error (every litellm error is an OpenAIError;
                # a 429 under 4 parallel calls), sandbox error, or failed
                # command (run_in raises RuntimeError) must not end the whole run.
                print(f"    attempt {i} failed ({type(err).__name__}); counted as reward 0")
                results.append(
                    self._failed(variant_id, i, f"attempt failed: {type(err).__name__}: {err}", 0.0)
                )
        pool.shutdown(wait=False, cancel_futures=True)
        self.sweep()
        return results

    @staticmethod
    def _failed(variant_id: str, index: int, output: str, seconds: float) -> dict:
        """The result of an attempt that produced no answer: reward 0."""
        return {
            "variant": variant_id,
            "index": index,
            "failing_before": [],
            "seeded": env.seed(variant_id)[env.SOURCE_FILE],
            "prompt": "",
            "answer": "",
            "output": output,
            "reward": 0,
            "seconds": seconds,
            "timed_out": False,
        }

    @classmethod
    def _timed_out(cls, variant_id: str, index: int, seconds: float) -> dict:
        """An attempt the infrastructure did not finish in time: reward 0, reported apart."""
        return {**cls._failed(variant_id, index, "attempt timed out", seconds), "timed_out": True}

    def sweep(self) -> None:
        """Delete rollout sandboxes of this run that an attempt did not get to delete.

        Each delete stands alone: the sweep races the attempts' own deletes, so
        a sandbox already on its way out must not end the sweep for the rest.
        """
        try:
            leftovers = list(
                self.daytona.list(
                    ListSandboxesQuery(labels={"demo": "rl", "run": self.run_id, "role": "rollout"})
                )
            )
        except DaytonaError as err:
            print(f"    sweep failed to list sandboxes: {err}", file=sys.stderr)
            return
        for sandbox in leftovers:
            state = getattr(sandbox.state, "value", sandbox.state)
            if state in ("destroying", "destroyed"):
                continue
            try:
                self.daytona.delete(sandbox)
                print(f"    swept leftover sandbox {sandbox.name}")
            except DaytonaError as err:
                print(f"    could not sweep {sandbox.name}: {err}", file=sys.stderr)

    def _create(self, name: str) -> Sandbox:
        """A create that retries: at scale a small share of creates fail (the cookbook saw ~1%)."""
        for attempt in range(4):
            try:
                return self.daytona.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=ENV_SNAPSHOT,
                        name=f"{name}-{secrets.token_hex(2)}",
                        labels=labels(self.run_id, "rollout"),
                        auto_stop_interval=10,
                    ),
                    timeout=120,
                )
            except DaytonaError as err:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
                print(f"    create retry ({type(err).__name__})", flush=True)
        raise AssertionError("unreachable")


def run_in_quiet(sandbox: Sandbox, command: str, cwd: str | None = None) -> str:
    response = sandbox.process.exec(command, cwd=cwd, timeout=120)
    return response.result or ""


def grade(sandbox: Sandbox, ws: str) -> str:
    return run_in_quiet(sandbox, env.GRADE_COMMAND, cwd=ws)


def policy(prompt: str) -> str:
    """The policy under training: one completion from the configured model."""
    response = litellm.completion(
        model=POLICY_MODEL,
        api_key=os.environ.get("LLM_API_KEY"),
        timeout=120,
        messages=[
            {
                "role": "system",
                "content": "You are a careful Python engineer fixing a small library.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def changed_lines(before: str, after: str, limit: int = 12) -> str:
    """Lines the attempt added or altered, so memory holds the fix rather than the whole file."""
    old = set(before.splitlines())
    new = [line for line in after.splitlines() if line.strip() and line not in old]
    return "\n".join(new[:limit]) + ("\n…" if len(new) > limit else "")


def qa_entry(result: dict) -> dict:
    """The scored memory entry for one attempt: task, what was changed, what the tests said."""
    reward = result["reward"]
    variant = env.BY_ID[result["variant"]]
    verdict = (
        "reward 1: every hidden test passed."
        if reward
        else "reward 0: hidden tests still fail:\n" + env.failure_lines(result["output"])
    )
    return {
        "type": "qa",
        "question": (
            f"Task {variant.id} ({env.FAMILIES[variant.family]}): fix ledger.py so the hidden "
            f"tests pass. Failing before the attempt: {', '.join(result['failing_before'])}."
        ),
        "answer": (
            f"Attempt changed these lines in ledger.py:\n"
            f"{changed_lines(result['seeded'], result['answer'])}\n{verdict}"
        ),
        "context": env.failure_lines(result["output"]) or result["output"][-400:],
        "feedback_text": verdict,
        "feedback_score": 5 if reward else 1,
    }


def recall_lessons(memory: Memory, variant_id: str) -> str:
    family = env.BY_ID[variant_id].family
    return memory.recall(
        f"Earlier attempts at {env.FAMILIES[family]} in ledger.py: what did the failing "
        "assertions expect (values, exception types, exact messages), and which changes passed?"
    )


def run_group(
    rollouts: Rollouts, memory: Memory | None, variant_id: str, size: int, session: str | None
) -> dict:
    """``size`` attempts of one variant in parallel sandboxes; scored into memory when given."""
    lessons = recall_lessons(memory, variant_id) if memory is not None else ""
    family = env.BY_ID[variant_id].family
    preview = " ".join(lessons.split())[:150]
    print(
        f"    [{variant_id} {family}] lessons recalled: "
        + (f"{len(lessons)} chars: {preview}…" if lessons else "none yet"),
        flush=True,
    )
    results = rollouts.group(variant_id, size, lessons)
    rewards = [r["reward"] for r in results]
    passes = sum(rewards)
    timed_out = sum(r["timed_out"] for r in results)
    learnable = 0 < passes < size
    mean_s = statistics.mean(r["seconds"] for r in results)
    print(
        f"    [{variant_id} {family}] rewards {' '.join(map(str, rewards))}  pass {passes}/{size}"
        f"{timed_out_note(timed_out)}  "
        f"{'mixed (learnable)' if learnable else 'uniform (no gradient)'}  {mean_s:.0f}s/attempt",
        flush=True,
    )
    if memory is not None and session is not None:
        for r in results:
            if r["prompt"]:  # not a timed-out placeholder
                memory.remember(qa_entry(r), session)
    return {
        "variant": variant_id,
        "family": family,
        "rewards": rewards,
        "timed_out": timed_out,
        "learnable": learnable,
        "lessons": lessons,
        "results": results,
    }


def timed_out_note(count: int) -> str:
    """Timed-out attempts are scored 0 but are an infrastructure stall, not the policy's
    failure: say how many next to a pass count, so a rate is not misread. Empty when none."""
    return f" ({count} timed out)" if count else ""


def run_improve(memory: Memory, session: str, step: int) -> None:
    print(f"=== improve after step {step} (memory sandbox) ===")
    for stage in memory.improve([session]):
        counts = ", ".join(f"{k}={v}" for k, v in (stage.get("counts") or {}).items())
        reason = f" ({stage['reason']})" if stage.get("reason") else ""
        print(f"    {stage['stage']}: {stage['status']}{reason} {counts}".rstrip())


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=len(env.STEPS))
    parser.add_argument("--group", type=int, default=8, help="attempts per variant per step")
    parser.add_argument("--parallel", type=int, default=4, help="rollout sandboxes at once")
    parser.add_argument("--eval-attempts", type=int, default=4, help="per held-out variant")
    parser.add_argument("--keep", action="store_true", help="keep the memory sandbox")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the cognee snapshot")
    args = parser.parse_args()

    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is required (https://app.daytona.io)")
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("LLM_API_KEY is required: the policy runs on the host")

    daytona = Daytona()
    ensure_secret(daytona)
    ensure_snapshot(daytona, rebuild=args.rebuild)
    ensure_env_snapshot(daytona)
    run_id = secrets.token_hex(3)
    print(f"=== run {run_id} ===")

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    memory_sandbox: Sandbox | None = None
    report: dict = {"run": run_id, "steps": [], "evaluation": {}}
    try:
        memory_sandbox, memory = start_memory_sandbox(daytona, run_id)
        rollouts = Rollouts(daytona, run_id, args.parallel)

        for step in range(1, args.steps + 1):
            variants = env.STEPS[(step - 1) % len(env.STEPS)]
            session = f"rl-{run_id}-step-{step}"
            print(
                f"=== step {step}/{args.steps}: variants {','.join(variants)} "
                f"(group {args.group}, parallel {args.parallel}) ==="
            )
            groups = [run_group(rollouts, memory, v, args.group, session) for v in variants]
            total = sum(sum(g["rewards"]) for g in groups)
            print(
                f"    step {step}: pass {total}/{args.group * len(variants)}"
                f"{timed_out_note(sum(g['timed_out'] for g in groups))}, "
                f"{sum(g['learnable'] for g in groups)}/{len(groups)} groups learnable"
            )
            report["steps"].append({"step": step, "session": session, "groups": groups})
            run_improve(memory, session, step)

        print(
            f"=== evaluation: held-out variants {','.join(env.HELD_OUT)} "
            f"({args.eval_attempts} attempts each, memory off then on) ==="
        )
        evaluation = {}
        for condition, mem in (("memory off", None), ("memory on", memory)):
            groups = [run_group(rollouts, mem, v, args.eval_attempts, None) for v in env.HELD_OUT]
            total = sum(sum(g["rewards"]) for g in groups)
            timed_out = sum(g["timed_out"] for g in groups)
            evaluation[condition] = {"groups": groups, "total": total, "timed_out": timed_out}
            print(
                f"    {condition}: pass {total}/{args.eval_attempts * len(env.HELD_OUT)}"
                f"{timed_out_note(timed_out)}"
            )
        report["evaluation"] = evaluation

        (out_dir / f"report-{run_id}.json").write_text(json.dumps(report, indent=2))

        print("=== exporting memory (COGX) ===")
        home = memory_sandbox.get_user_home_dir()
        # The embedded graph database is opened by a cognee database worker
        # process and holds a file lock; the export is a new process, so every
        # Python process in the sandbox is stopped first (the run is over anyway).
        # The slim image has no procps (no pkill/ps): walk /proc, kill is a builtin.
        run_in(memory_sandbox, KILL_PYTHON + "; sleep 2")
        print("    api server and its database workers stopped; exporting from the sandbox's state")
        export_cmd = (
            'python3 -c "import asyncio, cognee, os; from pathlib import Path; '
            "from cognee.modules.migration.archive import pack_archive; "
            f"r = asyncio.run(cognee.export('{DATASET}', format='cogx', destination='/tmp/rl.cogx')); "
            "p = pack_archive('/tmp/rl.cogx', '/tmp/rl.cogx.tar.gz'); "
            "print(f'[cogx] exported {r.dataset_name}: {r.num_nodes} nodes, {r.num_edges} edges, "
            "' + str(p.stat().st_size // 1000) + ' kB')\""
        )
        env_exports = " ".join(
            f"{k}={v}"
            for k, v in {
                "DATA_ROOT_DIRECTORY": f"{home}/memory-state/data",
                "SYSTEM_ROOT_DIRECTORY": f"{home}/memory-state/system",
                "ENABLE_BACKEND_ACCESS_CONTROL": "false",
                "TELEMETRY_DISABLED": "1",
                "LOG_LEVEL": "ERROR",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }.items()
        )
        run_in(memory_sandbox, f"env {env_exports} {export_cmd}", timeout=600)
        archive = memory_sandbox.fs.download_file("/tmp/rl.cogx.tar.gz") or b""
        (out_dir / "rl_memory.cogx.tar.gz").write_bytes(archive)
        print(f"    pulled to {out_dir / 'rl_memory.cogx.tar.gz'}")

        print("=== results ===")
        mean_attempt = statistics.mean(rollouts.timings) if rollouts.timings else 0
        for s in report["steps"]:
            n = args.group * len(s["groups"])
            print(
                f"    step {s['step']}: pass {sum(sum(g['rewards']) for g in s['groups'])}/{n}"
                f"{timed_out_note(sum(g['timed_out'] for g in s['groups']))}, "
                f"learnable groups {sum(g['learnable'] for g in s['groups'])}/{len(s['groups'])}"
            )
        n_eval = args.eval_attempts * len(env.HELD_OUT)
        for condition, data in evaluation.items():
            per = " ".join(
                f"{g['variant']} {sum(g['rewards'])}/{args.eval_attempts}" for g in data["groups"]
            )
            print(
                f"    held-out {condition}: {data['total']}/{n_eval}"
                f"{timed_out_note(data['timed_out'])}  ({per})"
            )
        print(f"    rollouts: {len(rollouts.timings)}, {mean_attempt:.0f}s per attempt")
        print(f"    report: {out_dir / f'report-{run_id}.json'}")
    finally:
        leftovers = []
        try:
            leftovers = daytona.list(ListSandboxesQuery(labels={"demo": "rl", "run": run_id}))
        except DaytonaError as err:
            print(f"could not list sandboxes of run {run_id}: {err}", file=sys.stderr)
        for sandbox in leftovers:
            if args.keep and memory_sandbox is not None and sandbox.id == memory_sandbox.id:
                print(f"\nMemory sandbox kept: {sandbox.name}")
                continue
            try:
                daytona.delete(sandbox)
            except DaytonaError as err:
                print(f"could not delete {sandbox.name}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
