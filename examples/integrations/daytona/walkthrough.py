"""Annotated terminal walkthrough of the Daytona memory handover.

Shows the output of ``handover.py`` one block at a time and, after each block,
a short note on what just happened. The notes are derived from the lines
themselves (status codes, archive sizes, dataset ids, durations), so they
describe the run on screen.

    python walkthrough.py                           # replay transcript.txt (no account needed)
    python walkthrough.py --pause                   # wait for Enter between blocks (presenting)
    python walkthrough.py --live                    # run handover.py now, annotate as it streams
    python walkthrough.py --live --record transcript.txt -- --keep   # save it, pass flags through
    python walkthrough.py --markdown > WALKTHROUGH.md                # the same as a document

A transcript is the stdout+stderr of ``handover.py`` with an elapsed-time
stamp in front of every line (``[  12.3s] ...``); ``--live --record`` writes
one, and ``python handover.py 2>&1 | python walkthrough.py --stamp`` also does.
"""

from __future__ import annotations

import argparse
import itertools
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TRANSCRIPT = HERE / "transcript.txt"
HANDOVER = HERE / "handover.py"

STAMP = re.compile(r"^\[\s*([\d.]+)s\] (.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Lines that open a new block of output.
HEADER = re.compile(
    r"^(=== .* ===|Handover round trip passed.*|Sandboxes kept:.*|"
    r"Traceback \(most recent call last\):|could not delete .*|sandboxes still running:.*)$"
)


@dataclass
class Block:
    header: str
    lines: list[tuple[float | None, str]] = field(default_factory=list)
    next_start: float | None = None

    @property
    def start(self) -> float | None:
        return self.lines[0][0] if self.lines else None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.next_start is None:
            return None
        return self.next_start - self.start

    @property
    def text(self) -> str:
        return "\n".join(text for _, text in self.lines)

    def find(self, pattern: str) -> re.Match[str] | None:
        return re.search(pattern, self.text, re.MULTILINE)


# --------------------------------------------------------------------------- parsing


def split_line(raw: str) -> tuple[float | None, str]:
    raw = ANSI.sub("", raw.rstrip("\n"))
    m = STAMP.match(raw)
    if m:
        return float(m.group(1)), m.group(2)
    return None, raw


def parse(lines) -> list[Block]:
    """Group stamped lines into blocks; a block starts at every HEADER line."""
    blocks: list[Block] = []
    current: Block | None = None
    for raw in lines:
        stamp, text = split_line(raw)
        if text.startswith("exit="):
            continue  # the recorder's trailer
        if HEADER.match(text) or current is None:
            current = Block(header=text if HEADER.match(text) else "(output before the first step)")
            blocks.append(current)
        current.lines.append((stamp, text))
    for a, b in itertools.pairwise(blocks):
        a.next_start = b.start
    return blocks


# --------------------------------------------------------------------------- notes

RL_PREAMBLE = [
    (
        "An RL rollout loop in the shape of the HUD cookbook: a bug-fix task with variants, one "
        "fresh Daytona sandbox per attempt, pytest as the grader, a binary reward, groups of "
        "attempts per variant."
    ),
    (
        "What the cookbook leaves ephemeral, this keeps: one memory sandbox runs cognee for the "
        "whole run. The agent recalls lessons before each group and writes every scored attempt "
        "back; improve() turns a step's attempts into lessons for the next. Held-out variants are "
        "then tried with memory off and on."
    ),
    "The number in front of each line is seconds since the run started.",
]

PREAMBLE = [
    (
        "Three Daytona sandboxes, one per phase, all from one snapshot with cognee installed. "
        "A supervisor agent writes memory, a worker agent in a second sandbox reads it and "
        "reports back, the supervisor reads the report in a third."
    ),
    (
        "Memory is cognee's embedded state on the sandbox disk. The two agents are two cognee "
        "users; cognee's ACLs decide what each can read. Between sandboxes travel one dataset as "
        "a COGX archive and a JSON token with its id."
    ),
    "The number in front of each line is seconds since handover.py started.",
]


# Blocks are explained in order, so the handover mode seen so far is known.
_seen = {"cogx": False}


def _secs(block: Block) -> str:
    d = block.duration
    if d is None:
        return ""
    if d < 1:
        return " Under a second."
    if d < 90:
        return f" {d:.0f}s."
    return f" {d / 60:.1f} min."


def explain_snapshot(block: Block) -> list[str]:
    rollout_image = "cognee-rl-env" in block.header
    if "exists, reusing" in block.header:
        note = (
            "The rollout image (Debian, Python 3.12, pytest; 1 vCPU, 1 GB) exists and is active."
            if rollout_image
            else (
                "The snapshot (Debian, Python 3.12, pip install cognee) exists and is active, so "
                "no sandbox has to install anything."
            )
        )
    elif "activating" in block.header:
        note = (
            "Daytona had deactivated the unused snapshot and refuses to start sandboxes from it. "
            "handover.py reactivated it, which is a rebuild, and waited through the states shown."
        )
    else:
        note = "First run: Daytona builds the snapshot from the image spec. Happens once."
    return [note + _secs(block)]


def explain_secret(block: Block) -> list[str]:
    return [
        (
            "First run only: the LLM key is stored as a Daytona Secret scoped to api.openai.com. "
            "Sandboxes get a placeholder; the proxy swaps in the real key for that host only."
        )
    ]


def explain_run(block: Block) -> list[str]:
    m = re.search(r"run (\w+)", block.header)
    run_id = m.group(1) if m else "<run>"
    return [
        (
            f"{run_id} is this run's id. It is in every sandbox name and in a run= label, which "
            "the cleanup lists by, so nothing created here is left behind."
        )
    ]


def explain_create(block: Block) -> list[str]:
    m = re.search(r"creating sandbox: cognee-(\w+)-", block.header)
    role = m.group(1) if m else "?"
    if role != "supervisor":
        return [
            f"A fresh sandbox for the {role} phase: same snapshot, same posture (Secret binding, "
            "labels, auto-stop). It starts empty; the next block fills it." + _secs(block)
        ]
    out = [
        (
            "The first sandbox, asked for with the Secret bound to LLM_API_KEY and an allow list of "
            "two hosts."
        )
    ]
    if "tier sets network policy" in block.text:
        out.append(
            "Daytona rejected the allow list: this organization is Tier 1/2, where network "
            "policy is set org-wide. The sandbox runs behind that policy instead (essential "
            "services only, enforced on direct connections)."
        )
    out.append("Then the payload and the COGX helper were uploaded. No memory yet." + _secs(block))
    return out


def explain_phase_brief(block: Block) -> list[str]:
    m = block.find(r'"dataset_id": "([0-9a-f-]{36})"')
    ds = m.group(1) if m else "<uuid>"
    return [
        (
            "The supervisor remembers a private note and a handover briefing, in two datasets of its "
            "own. Each remember() runs cognify: LLM entity extraction through the proxy, where the "
            "key is substituted. That is where the time goes."
        ),
        f"Then it grants the worker read and write on the briefing dataset and writes its id "
        f"({ds[:8]}…) into the token. Ids are the only cross-user address; names are per user."
        + _secs(block),
    ]


def explain_security(block: Block) -> list[str]:
    out = []
    if "is the placeholder" in block.text:
        out.append("Inside the sandbox LLM_API_KEY is the placeholder, not a key.")
    elif "NOT the placeholder" in block.text:
        out.append("The value inside the sandbox is not the placeholder: Secret binding failed.")
    proxy = block.find(r"via proxy path: (\d+)")
    direct = block.find(r"via direct path \(proxy env ignored\): (\d+|blocked \w+)")
    if proxy and proxy.group(1) == "200" and direct:
        out.append(
            f"The same placeholder gets 200 from OpenAI through the proxy and {direct.group(1)} "
            "when the proxy is bypassed: the key exists only in transit."
        )
    if "WARNING" in block.text:
        out.append(
            "example.com is blocked on a direct connection but reachable through the proxy. On "
            "Tier 1/2 Daytona polices direct connections only. Reported, not fatal; "
            "--strict-walls would fail here." + _secs(block)
        )
    elif "example.com" in block.text:
        out.append("example.com is unreachable both ways." + _secs(block))
    return out


def explain_fork(block: Block) -> list[str]:
    if "fork not available" in block.text:
        return [
            "A fork (copy-on-write clone) was tried first; Daytona only forks Linux VM "
            "sandboxes and this is a container. Falling back to a copy." + _secs(block)
        ]
    return ["The sandbox was forked; the clone has the parent's disk." + _secs(block)]


def explain_copy(block: Block) -> list[str]:
    size = block.find(r"state archive: ([\d.]+ MB)")
    return [
        "Raw-state handover: the whole ~/cognee-state directory (SQLite, graph files, vector "
        f"stores, every dataset) tarred ({size.group(1) if size else '?'}), pulled to the host, "
        "parent stopped. The next sandbox gets the archive." + _secs(block)
    ]


def explain_export(block: Block) -> list[str]:
    _seen["cogx"] = True
    counts = block.find(r"(\d+) nodes, (\d+) edges")
    size = block.find(r"\((\d+) kB\)")
    files = block.find(r"archive files: (.*)")
    out = [
        "The handover. cognee.export(format='cogx', include_permissions=True) on the shared "
        "dataset, as its owner"
        + (f": {counts.group(1)} nodes, {counts.group(2)} edges" if counts else "")
        + (f", {size.group(1)} kB packed." if size else ".")
    ]
    if files:
        out.append(
            f"The archive is plain files: {files.group(1)}. permissions.json carries the owner "
            "and the worker's grants. The private dataset is not in it and stays behind."
        )
    out[-1] += " Parent stopped once the archive was pulled." + _secs(block)
    return out


def explain_import(block: Block) -> list[str]:
    counts = block.find(r"restored dataset '(\w+)'.*?: (\d+) nodes, (\d+) edges")
    reissued = block.find(r"dataset ([0-9a-f-]{8})[0-9a-f-]* is ([0-9a-f-]{8})")
    out = []
    if counts:
        out.append(
            f"cognee.remember(COGXArchiveSource) restored '{counts.group(1)}' with "
            f"{counts.group(2)} nodes and {counts.group(3)} edges, no LLM call; the time is "
            "the embeddings being recomputed. The owner and the granted users were recreated "
            "from permissions.json, grants included."
        )
    if "default_user@example.com" in block.text:
        out.append(
            "The default_user grant is the previous sandbox's importer keeping read access, as "
            "cognee's import always does; it travelled along in permissions.json."
        )
    if reissued:
        out.append(
            f"Restored accounts get new ids and a dataset id derives from its owner's, so the "
            f"dataset is {reissued.group(2)}… here instead of {reissued.group(1)}…. The token "
            "was rewritten to match." + _secs(block)
        )
    return out


def explain_phase_work(block: Block) -> list[str]:
    out = []
    if "raised (Status code:" in block.text:
        out.append(
            "The two error lines at the top are cognee logging the denials provoked below; "
            "stderr arrives before the buffered prints."
        )
    if "LiteLLM.Info" in block.text:
        out.append("The litellm banner is noise from a non-fatal internal error.")
    private = (
        "the private dataset id is denied (it does not even exist here)"
        if _seen["cogx"]
        else "the private dataset, present on this disk, is denied by the ACLs"
    )
    out.append(
        "The worker recalls the briefing by dataset id and gets an answer from the graph that "
        f"arrived. Its two other attempts fail as they should: {private}, and the name "
        "'handover' resolves to nothing, because names are per user."
    )
    if "report stored" in block.text:
        out.append(
            "Then it writes its report into the shared dataset by id; another cognify run."
            + _secs(block)
        )
    return out


def explain_phase_review(block: Block) -> list[str]:
    return [
        "Back as the supervisor, in the third sandbox: the report the worker wrote in the second "
        "one comes back from a recall. Round trip closed; the private note never moved."
        + _secs(block)
    ]


def explain_summary(block: Block) -> list[str]:
    mode = block.find(r"Handover:\s+(\w+)")
    ds = block.find(r"Dataset shared by UUID:\s+(\S+)")
    out = []
    if mode:
        out.append(
            f"Handover mode {mode.group(1)}, network wall and warnings as above, and the id of the "
            "one dataset that crossed"
            + (f" ({ds.group(1)[:8]}…, as re-issued in the last sandbox)." if ds else ".")
        )
    out.append("Cleanup deleted the run's sandboxes, children first. Snapshot and Secret stay.")
    return out


def explain_kept(block: Block) -> list[str]:
    return ["--keep: the sandboxes stay stopped; the one-liner deletes them by label."]


def explain_failure(block: Block) -> list[str]:
    out = ["The run failed here."]
    if "is inactive" in block.text:
        out.append("Cause: the snapshot was inactive; current handover.py reactivates it first.")
    if "isolation breached" in block.text:
        out.append("Cause: a wall did not hold, so no memory was handed over.")
    out.append("The sandboxes of this run were still swept and deleted.")
    return out


def explain_leak(block: Block) -> list[str]:
    return ["A sandbox could not be deleted; the run exits non-zero. Delete it by its run label."]


def explain_memory_sandbox(block: Block) -> list[str]:
    m = block.find(r"healthy after (\d+)s")
    return [
        "The one long-lived sandbox: the cognee API server from the cognee snapshot, with the LLM "
        "key as a Secret placeholder. Private sandbox; the host reaches it through the preview "
        "URL with its token." + (f" Up in {m.group(1)}s." if m else "")
    ]


def explain_step(block: Block) -> list[str]:
    m = re.search(r"step (\d+)/(\d+): variants (\S+) \(group (\d+)", block.header)
    out = []
    if m:
        out.append(
            f"Training step {m.group(1)} of {m.group(2)}: one variant per bug family, "
            f"{m.group(4)} attempts each, every attempt in its own fresh sandbox. Before a group, "
            "one recall fetches the family's lessons; 'none yet' means the memory is still empty."
        )
    # rl_loop appends " (N timed out)" to a pass count when attempts timed out.
    groups = re.findall(
        r"\[(\w+) (\w+)\] rewards ([01 ]+)  pass (\d+)/(\d+)(?: \(\d+ timed out\))?  "
        r"(mixed|uniform)",
        block.text,
    )
    if groups:
        mixed = [g[0] for g in groups if g[5] == "mixed"]
        uniform = [g[0] for g in groups if g[5] == "uniform"]
        out.append(
            "Rewards are 1 when every hidden test passes, else 0. A group with mixed rewards is "
            "one a group-relative trainer can learn from; a uniform group gives it nothing."
            + (f" Mixed here: {', '.join(mixed)}." if mixed else " No mixed group here.")
            + (f" Uniform: {', '.join(uniform)}." if uniform else "")
        )
    m = block.find(
        r"step \d+: pass (\d+)/(\d+)(?: \(\d+ timed out\))?, (\d+)/(\d+) groups learnable"
    )
    if m:
        out.append(
            f"Step total {m.group(1)}/{m.group(2)} passed, {m.group(3)}/{m.group(4)} groups "
            "learnable. Each attempt is now in memory as a scored entry: the task, the answer, "
            "and the failing assertions when it failed." + _secs(block)
        )
    return out


def explain_rl_improve(block: Block) -> list[str]:
    qa = block.find(r"persist_session_qa: (\w+)")
    skipped = re.findall(r"(\w+): skipped", block.text)
    return [
        "improve() over the step's session, inside the memory sandbox. Feedback weights apply "
        "the scores, and the scored attempts are persisted into the graph"
        + (f" ({qa.group(1)})" if qa else "")
        + ". The lesson-drafting stages are off here (they hang on per-turn feedback analysis), "
        "so the lessons the next step sees are synthesized at recall time from these persisted "
        "attempts: their changes, verdicts and failing assertions."
        + (f" Skipped: {', '.join(skipped)}." if skipped else "")
        + _secs(block)
    ]


def explain_eval(block: Block) -> list[str]:
    off = block.find(r"memory off: pass (\d+)/(\d+)")
    on = block.find(r"memory on: pass (\d+)/(\d+)")
    out = [
        (
            "Held-out variants: same families, bugs never seen in training. First with no memory, "
            "then with the lessons recalled."
        )
    ]
    if off and on:
        out.append(
            f"Memory off {off.group(1)}/{off.group(2)}, memory on {on.group(1)}/{on.group(2)}. "
            "That difference is what memory alone buys the policy, with no weights changed."
            + _secs(block)
        )
    return out


def explain_rl_export(block: Block) -> list[str]:
    m = block.find(r"exported \w+: (\d+) nodes, (\d+) edges, (\d+) kB")
    return [
        "The memory checkpoint: cognee.export(format='cogx') of the rl_memory dataset"
        + (f", {m.group(1)} nodes and {m.group(2)} edges in {m.group(3)} kB" if m else "")
        + ", pulled to the host. Restore it into another memory sandbox to continue training "
        "from here." + _secs(block)
    ]


def explain_rl_results(block: Block) -> list[str]:
    return [
        (
            "Per step: pass rate and learnable groups. Held-out: memory off against memory on. The "
            "report JSON has every attempt. Cleanup then deletes the run's sandboxes by label."
        )
    ]


EXPLAINERS = (
    (r"^=== memory sandbox: ", explain_memory_sandbox),
    (r"^=== step \d+/\d+: ", explain_step),
    (r"^=== improve after step ", explain_rl_improve),
    (r"^=== evaluation: ", explain_eval),
    (r"^=== exporting memory ", explain_rl_export),
    (r"^=== results ===$", explain_rl_results),
    (r"^=== snapshot ", explain_snapshot),
    (r"^=== building snapshot ", explain_snapshot),
    (r"^=== creating Secret ", explain_secret),
    (r"^=== run \w+ ===$", explain_run),
    (r"^=== creating sandbox: ", explain_create),
    (r"\(phase: brief\) ===$", explain_phase_brief),
    (r"^=== security check inside ", explain_security),
    (r"^=== forking ", explain_fork),
    (r"^=== copying ", explain_copy),
    (r"^=== exporting ", explain_export),
    (r"^=== importing into ", explain_import),
    (r"\(phase: work\) ===$", explain_phase_work),
    (r"\(phase: review\) ===$", explain_phase_review),
    (r"^Handover round trip passed", explain_summary),
    (r"^Sandboxes kept:", explain_kept),
    (r"^Traceback ", explain_failure),
    (r"^(could not delete|sandboxes still running)", explain_leak),
)


def explain(block: Block) -> list[str]:
    for pattern, fn in EXPLAINERS:
        if re.search(pattern, block.header):
            return [p for p in fn(block) if p]
    return []


# --------------------------------------------------------------------------- rendering


class Term:
    def __init__(self, color: bool, delay: float, pause: bool):
        self.color = color
        self.delay = delay
        self.pause = pause
        self.width = min(shutil.get_terminal_size((100, 24)).columns, 100)

    def c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def line(self, stamp: float | None, text: str) -> None:
        prefix = self.c("2", f"[{stamp:7.1f}s] ") if stamp is not None else ""
        body = self.c("1;36", text) if HEADER.match(text) else text
        print(prefix + body, flush=True)
        if self.delay:
            time.sleep(self.delay)

    def note(self, paragraphs: list[str], title: str = "what happened") -> None:
        if not paragraphs:
            return
        bar = self.c("33", "  │ ")
        print()
        print(self.c("1;33", f"  ▶ {title}"))
        for p in paragraphs:
            for wrapped in textwrap.wrap(p, self.width - 6):
                print(bar + wrapped)
        print(flush=True)
        if self.pause:
            try:
                input(self.c("2", "  — Enter for the next step — "))
            except EOFError:
                pass

    def block(self, block: Block) -> None:
        for stamp, text in block.lines:
            self.line(stamp, text)
        self.note(explain(block))


def render_markdown(blocks: list[Block]) -> str:
    rl = _seen.get("rl")
    out = [
        "# Daytona RL rollouts with cognee memory, annotated"
        if rl
        else "# Daytona memory handover, annotated",
        "",
    ]
    out += [p + "\n" for p in preamble_for(blocks)]
    for block in blocks:
        title = block.header.strip("= ").strip()
        out += [f"## {title}", "", "```text"]
        for stamp, text in block.lines:
            out.append(((f"[{stamp:7.1f}s] " if stamp is not None else "") + text).rstrip())
        out += ["```", ""]
        out += [p + "\n" for p in explain(block)]
    # One final newline (print adds it): what the repo's end-of-file hook expects.
    return "\n".join(out).rstrip("\n")


# --------------------------------------------------------------------------- sources


def live_lines(script: Path, extra: list[str], record: Path | None):
    """Run a script now; yield its lines with elapsed-time stamps, optionally recording them."""
    script = script.resolve()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script), *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=script.parent,
        # Its own session, so a terminal Ctrl-C reaches only this process and
        # the script gets exactly the one SIGINT forwarded below. In the
        # terminal's process group it would get a second one from us while
        # its finally is deleting sandboxes, and abort that cleanup.
        start_new_session=True,
    )
    t0 = time.time()
    sink = record.open("w") if record else None
    interrupts = 0
    try:
        assert proc.stdout is not None
        lines = iter(proc.stdout)
        while True:
            try:
                raw = next(lines)
            except StopIteration:
                break
            except KeyboardInterrupt:
                # First Ctrl-C: let the script clean up (its finally deletes
                # sandboxes) and keep showing what it prints while it does.
                # A second one is the user insisting: stop it outright.
                interrupts += 1
                if interrupts == 1:
                    proc.send_signal(signal.SIGINT)
                else:
                    proc.terminate()
                continue
            # Trailing spaces stripped so a recorded transcript passes the
            # repo's trailing-whitespace hook as written.
            stamped = f"[{time.time() - t0:7.1f}s] {ANSI.sub('', raw).rstrip()}\n"
            if sink:
                sink.write(stamped)
                sink.flush()
            yield stamped
        code = proc.wait()
        if sink:
            sink.write(f"exit={code}\n")
        yield f"exit={code}"
    finally:
        if sink:
            sink.close()


def preamble_for(blocks_or_lines) -> list[str]:
    """The RL preamble when the run has a memory sandbox block, else the handover one."""
    return RL_PREAMBLE if _seen.get("rl") else PREAMBLE


def stream(lines, term: Term, rl: bool = False) -> int:
    """Annotate a stream as it arrives: a block is explained when the next one starts."""
    _seen["rl"] = rl
    term.note(preamble_for(None), title="what you are about to see")
    current: Block | None = None
    code = 0
    for raw in lines:
        stamp, text = split_line(raw)
        if text.startswith("exit="):
            code = int(text[5:] or 0)
            continue
        if HEADER.match(text) or current is None:
            if current is not None:
                current.next_start = stamp
                term.note(explain(current))
            current = Block(header=text if HEADER.match(text) else "(output before the first step)")
        current.lines.append((stamp, text))
        term.line(stamp, text)
    if current is not None:
        term.note(explain(current))
    return code


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--live", action="store_true", help="run the script now and annotate it")
    parser.add_argument(
        "--script", type=Path, default=HANDOVER, help="what --live runs (default handover.py)"
    )
    parser.add_argument("--record", type=Path, help="with --live: write the transcript here")
    parser.add_argument("--stamp", action="store_true", help="annotate stdin (a live pipe)")
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown document")
    parser.add_argument("--pause", action="store_true", help="wait for Enter after each step")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between replayed lines")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("extra", nargs="*", help="arguments passed to handover.py after --")
    args = parser.parse_args()

    term = Term(color=not args.no_color and sys.stdout.isatty(), delay=args.delay, pause=args.pause)

    rl = "rl" in str(args.script if args.live else args.transcript).lower().split("/")[-1] or (
        "rl_loop" in str(args.script)
    )
    if args.live:
        raise SystemExit(stream(live_lines(args.script, args.extra, args.record), term, rl=rl))
    if args.stamp:
        t0 = time.time()
        stream((f"[{time.time() - t0:7.1f}s] {raw}" for raw in sys.stdin), term, rl=rl)
        return

    if not args.transcript.is_file():
        raise SystemExit(
            f"no transcript at {args.transcript}; record one with --live --record {args.transcript}"
        )
    blocks = parse(args.transcript.read_text().splitlines())
    _seen["rl"] = rl or any(b.header.startswith("=== memory sandbox") for b in blocks)
    if args.markdown:
        print(render_markdown(blocks))
        return
    term.note(preamble_for(blocks), title="what you are about to see")
    for block in blocks:
        term.block(block)


if __name__ == "__main__":
    main()
