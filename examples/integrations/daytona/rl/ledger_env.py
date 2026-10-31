"""The RL environment: a tiny billing ledger with seeded bugs and hidden tests.

Attempts are single-shot (one completion, then the grader), so an attempt
never sees the assertions itself; the only channel for that feedback is the
memory that carries it from earlier attempts to later ones.

Shaped like the HUD cookbook's ``fix_calc`` template: one task, many
variants. Each variant seeds one bug into ``ledger.py``; the agent is told
which tests fail (names only) and must return a fixed file; the reward is
binary — every hidden test passes, or not.

The hidden tests enforce conventions the task text does not state (money
rounds half up, a repeated idempotency key returns the original charge id,
retries are five attempts with doubling delays, refunds past the remaining
amount raise ``LedgerError("over_refund")``, …). An attempt only learns them
by failing, and the failure output is what gets written to memory. Variants
come in families that share a convention, so a lesson learned on one variant
transfers to its siblings and to the held-out variant of the same family.

Nothing here imports cognee or daytona: ``seed(variant)`` returns the files
to put in a sandbox, ``GRADE_COMMAND`` runs the tests, ``passed(output)``
reads the verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORKSPACE = "ws"
SOURCE_FILE = "ledger.py"
TEST_FILE = "test_ledger.py"
# Short tracebacks: the "E" lines carry the assertion, the expected regex, DID NOT RAISE.
GRADE_COMMAND = "python3 -m pytest -q --tb=short -p no:cacheprovider"

README = """\
ledger.py — a tiny billing ledger. All money is integer cents.

Ledger.charge(amount, idempotency_key) -> charge id (str)
Ledger.refund(charge_id, amount) -> cents still refundable on that charge
Ledger.balance() -> cents charged minus cents refunded, across all charges
apply_discount(cents, percent) -> discounted price in cents
split_evenly(cents, parts) -> list of `parts` integer shares summing to cents
with_retry(fn, attempts=5, base_delay=1.0, sleep=time.sleep) -> fn() result;
    retries on any exception, calling sleep(delay) between attempts
"""

REFERENCE = '''\
"""A tiny billing ledger. All money is integer cents."""


class LedgerError(Exception):
    pass


class Ledger:
    def __init__(self):
        self.charges = {}  # charge id -> {"amount": int, "key": str, "refunded": int}
        self.by_key = {}  # idempotency key -> charge id
        self._next = 1

    def charge(self, amount, idempotency_key):
        if amount <= 0:
            raise LedgerError("amount must be positive cents")
        if idempotency_key in self.by_key:
            existing = self.by_key[idempotency_key]
            if self.charges[existing]["amount"] != amount:
                raise LedgerError("idempotency_conflict")
            return existing
        charge_id = f"ch_{self._next}"
        self._next += 1
        self.charges[charge_id] = {"amount": amount, "key": idempotency_key, "refunded": 0}
        self.by_key[idempotency_key] = charge_id
        return charge_id

    def refund(self, charge_id, amount):
        charge = self.charges.get(charge_id)
        if charge is None:
            raise LedgerError("unknown charge")
        remaining = charge["amount"] - charge["refunded"]
        if amount > remaining:
            raise LedgerError("over_refund")
        charge["refunded"] += amount
        if charge["refunded"] == charge["amount"]:
            self.by_key.pop(charge["key"], None)  # a fully refunded charge releases its key
        return remaining - amount

    def balance(self):
        return sum(c["amount"] - c["refunded"] for c in self.charges.values())


def apply_discount(cents, percent):
    return (cents * (100 - percent) + 50) // 100


def split_evenly(cents, parts):
    base, remainder = divmod(cents, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def with_retry(fn, attempts=5, base_delay=1.0, sleep=None):
    if sleep is None:
        import time

        sleep = time.sleep
    delay = base_delay
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:
            if attempt == attempts - 1:
                raise LedgerError("retry_exhausted") from err
            sleep(delay)
            delay = delay * 2
'''

TESTS = """\
import pytest

from ledger import Ledger, LedgerError, apply_discount, split_evenly, with_retry


def test_discount_rounding():
    assert apply_discount(1000, 15) == 850
    assert apply_discount(999, 10) == 899


def test_discount_edge_cases():
    assert apply_discount(125, 10) == 113
    assert apply_discount(105, 50) == 53
    assert apply_discount(115, 50) == 58


def test_split_remainder():
    assert split_evenly(100, 3) == [34, 33, 33]
    assert split_evenly(7, 4) == [2, 2, 2, 1]


def test_charge_idempotency():
    ledger = Ledger()
    first = ledger.charge(500, "k1")
    assert ledger.charge(500, "k1") == first
    assert len(ledger.charges) == 1


def test_charge_distinct_keys():
    ledger = Ledger()
    a = ledger.charge(500, "k1")
    b = ledger.charge(500, "k2")
    assert a != b
    assert len(ledger.charges) == 2


def test_charge_repeat_returns_id():
    ledger = Ledger()
    first = ledger.charge(300, "order-9")
    again = ledger.charge(300, "order-9")
    assert again == first
    assert again in ledger.charges


def test_charge_key_reuse():
    ledger = Ledger()
    ledger.charge(300, "order-9")
    with pytest.raises(LedgerError, match="idempotency_conflict"):
        ledger.charge(301, "order-9")
    assert len(ledger.charges) == 1


def test_charge_validation():
    ledger = Ledger()
    with pytest.raises(LedgerError):
        ledger.charge(0, "k")


def test_refund_partial():
    ledger = Ledger()
    cid = ledger.charge(1000, "k")
    assert ledger.refund(cid, 300) == 700
    assert ledger.refund(cid, 300) == 400
    assert ledger.balance() == 400
    with pytest.raises(LedgerError, match="over_refund"):
        ledger.refund(cid, 500)


def test_refund_limit():
    ledger = Ledger()
    cid = ledger.charge(1000, "k")
    ledger.refund(cid, 600)
    with pytest.raises(LedgerError, match="over_refund"):
        ledger.refund(cid, 500)


def test_refund_complete():
    ledger = Ledger()
    first = ledger.charge(500, "k")
    assert ledger.refund(first, 500) == 0
    second = ledger.charge(500, "k")
    assert second != first
    assert ledger.balance() == 500


def test_balance():
    ledger = Ledger()
    a = ledger.charge(1000, "a")
    ledger.charge(250, "b")
    ledger.refund(a, 100)
    assert ledger.balance() == 1150


def test_retry_delays():
    sleeps = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise RuntimeError("boom")
        return "ok"

    assert with_retry(flaky, base_delay=1, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2, 4]


def test_retry_attempt_count():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(LedgerError):
        with_retry(always, base_delay=0, sleep=lambda s: None)
    assert calls["n"] == 5


def test_retry_gives_up():
    def always():
        raise ValueError("last")

    with pytest.raises(LedgerError, match="retry_exhausted") as info:
        with_retry(always, attempts=3, base_delay=0, sleep=lambda s: None)
    assert isinstance(info.value.__cause__, ValueError)
"""


@dataclass(frozen=True)
class Variant:
    id: str
    family: str
    old: str  # text in REFERENCE to replace
    new: str  # the seeded bug

    def source(self) -> str:
        assert REFERENCE.count(self.old) == 1, self.id
        return REFERENCE.replace(self.old, self.new)


FAMILIES = {
    "money_rounding": "discount arithmetic on integer cents",
    "idempotency": "repeated idempotency keys on charge()",
    "retries": "with_retry attempt count and backoff",
    "refunds": "refund limits and accumulation",
}

VARIANTS = [
    # money_rounding: round half up with integer math; every variant does it some other way
    Variant(
        "A1",
        "money_rounding",
        "    return (cents * (100 - percent) + 50) // 100",
        "    return int(cents * (100 - percent) / 100)",
    ),
    Variant(
        "A2",
        "money_rounding",
        "    return (cents * (100 - percent) + 50) // 100",
        "    return cents * (100 - percent) // 100",
    ),
    Variant(
        "A3",
        "money_rounding",
        "    return (cents * (100 - percent) + 50) // 100",
        "    return round(cents * (100 - percent) / 100)",
    ),
    # idempotency: same key + same amount -> original id; same key + other amount ->
    # LedgerError("idempotency_conflict"); every variant loses the conflict rule
    Variant(
        "B1",
        "idempotency",
        "        if idempotency_key in self.by_key:\n"
        "            existing = self.by_key[idempotency_key]\n"
        '            if self.charges[existing]["amount"] != amount:\n'
        '                raise LedgerError("idempotency_conflict")\n'
        "            return existing\n",
        "",
    ),
    Variant(
        "B2",
        "idempotency",
        '            if self.charges[existing]["amount"] != amount:\n'
        '                raise LedgerError("idempotency_conflict")\n',
        "",
    ),
    Variant(
        "B3",
        "idempotency",
        '                raise LedgerError("idempotency_conflict")',
        "                return self.charge(amount, idempotency_key + '-dup')",
    ),
    # retries: five attempts, doubling delays, then LedgerError("retry_exhausted") chained
    # from the last error; every variant gives up some other way
    Variant(
        "C1",
        "retries",
        '                raise LedgerError("retry_exhausted") from err',
        "                raise",
    ),
    Variant(
        "C2",
        "retries",
        '                raise LedgerError("retry_exhausted") from err',
        '                raise LedgerError(f"gave up after {attempts} attempts")',
    ),
    Variant(
        "C3",
        "retries",
        '                raise LedgerError("retry_exhausted") from err',
        "                return None",
    ),
    # refunds: refunds accumulate, past the remaining -> LedgerError("over_refund"), and a
    # fully refunded charge releases its key; every variant loses the key release
    Variant(
        "D1",
        "refunds",
        '        if charge["refunded"] == charge["amount"]:\n'
        '            self.by_key.pop(charge["key"], None)  # a fully refunded charge releases its key\n',
        "",
    ),
    Variant(
        "D2",
        "refunds",
        '        charge["refunded"] += amount\n'
        '        if charge["refunded"] == charge["amount"]:\n'
        '            self.by_key.pop(charge["key"], None)  # a fully refunded charge releases its key\n',
        '        charge["refunded"] = amount\n',
    ),
    Variant(
        "D3",
        "refunds",
        '            raise LedgerError("over_refund")\n'
        '        charge["refunded"] += amount\n'
        '        if charge["refunded"] == charge["amount"]:\n'
        '            self.by_key.pop(charge["key"], None)  # a fully refunded charge releases its key\n',
        '            raise LedgerError("insufficient funds")\n'
        '        charge["refunded"] += amount\n',
    ),
]
BY_ID = {v.id: v for v in VARIANTS}

# Training variants per step (one per family) and the held-out set for evaluation.
STEPS = [["A1", "B1", "C1", "D1"], ["A2", "B2", "C2", "D2"], ["A1", "B1", "C1", "D1"]]
HELD_OUT = ["A3", "B3", "C3", "D3"]


def seed(variant_id: str) -> dict[str, str]:
    """Files to put in a fresh sandbox for this variant: the buggy source and the hidden tests."""
    return {SOURCE_FILE: BY_ID[variant_id].source(), TEST_FILE: TESTS}


def failing_tests(output: str) -> list[str]:
    """Names of the failing tests in pytest's short summary, in order."""
    names = []
    for line in output.splitlines():
        if line.startswith("FAILED "):
            names.append(line.split("::", 1)[1].split(" ", 1)[0])
    return names


def passed(output: str) -> bool:
    """Binary reward: pytest reported no failures or errors and at least one pass."""
    last = output.strip().splitlines()[-1] if output.strip() else ""
    return " passed" in last and "failed" not in last and "error" not in last


def failure_lines(output: str, limit: int = 16) -> str:
    """What the failing tests expected: pytest's ``E`` lines, the FAILED summary, and the
    test's own source line from the short traceback (that line is what names an expected
    exception and message when the verdict is ``DID NOT RAISE``)."""
    keep: list[str] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("FAILED ", "E ")):
            keep.append(line.strip())
        elif re.match(rf"^{re.escape(TEST_FILE)}:\d+: in test_", line) and i + 1 < len(lines):
            keep.append(f"{line.split(' in ')[-1]}: {lines[i + 1].strip()}")
    return "\n".join(keep[:limit])


def task_prompt(variant_id: str, source: str, failing: list[str], lessons: str) -> str:
    """What the policy sees. The tests themselves are hidden; only their names are given."""
    lesson_block = (
        f"\nLessons recalled from memory about {FAMILIES[BY_ID[variant_id].family]}:\n{lessons}\n"
        if lessons
        else ""
    )
    return (
        f"`pytest` fails in `{WORKSPACE}`. Fix `{SOURCE_FILE}` so every test in `{TEST_FILE}` "
        f"passes. You cannot read or edit the tests. Failing tests: {', '.join(failing)}.\n\n"
        f"README:\n{README}\n{lesson_block}\nCurrent {SOURCE_FILE}:\n```python\n{source}```\n\n"
        f"Return the complete corrected {SOURCE_FILE} in one ```python block and nothing else."
    )


# Fence tags that mean "this is the module": python, python3 or py in any case,
# or none. Other languages (diff, text, bash) are not.
_PYTHON_TAGS = {"", "python", "python3", "py"}


def extract_source(answer: str) -> str | None:
    """The module from the model's answer: its longest python block, or None.

    Longest, not first: a model that shows a snippet or a diff before the full
    file would otherwise have the snippet graded as the whole module, and a
    parsing slip would be scored as the policy's failure. Fences are taken in
    pairs, so one block's closing fence is never read as the next one's opening.
    """
    blocks = []
    # Odd segments between fences are block contents: tag line, then body. With
    # an odd number of fences the last one is unclosed and its tail still counts,
    # as an answer that omits the closing fence would otherwise yield nothing.
    for segment in answer.split("```")[1::2]:
        tag, newline, body = segment.partition("\n")
        if newline and tag.strip().lower() in _PYTHON_TAGS:
            blocks.append(body)
    return max(blocks, key=len) if blocks else None
