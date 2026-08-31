"""LiteLLM-proxy budget exhaustion must be terminal, and nothing else may be.

A proxy spend-cap rejection cannot clear by retrying, but cognee used to treat
it as a transient rate limit: one logical LLM call fanned out to ~36 HTTP
requests over ~270s before failing.

The classification is unavoidably message-text based, because every structured
signal is destroyed on the way up:

* the error surfaces as ``InstructorRetryException``, whose ``__cause__`` chain
  is ``InstructorRetryException -> RetryError -> RateLimitError``;
* the outermost exception carries no ``status_code`` and no ``response``;
* the client-side ``RateLimitError`` is a *different class* from
  ``litellm.BudgetExceededError``, so an ``isinstance`` check cannot fire;
* that ``RateLimitError``'s ``.response.json()`` raises ``JSONDecodeError``
  because the body has already been consumed, so the proxy's structured
  ``error.type == "budget_exceeded"`` field is unreadable.

Text matching is therefore load-bearing and fragile across litellm upgrades,
which is what these tests pin. Two failure directions matter equally:

* a FALSE NEGATIVE resurrects the retry storm;
* a FALSE POSITIVE is worse — ``str(InstructorRetryException)`` embeds the
  model's own partial completion, so an over-eager match would turn a routine,
  retryable extraction failure into a hard 402 for any document that merely
  discusses an exceeded budget.
"""

import litellm
import pytest
from instructor.core.exceptions import FailedAttempt, InstructorRetryException
from tenacity import RetryError
from tenacity import Future as TenacityFuture

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    LLMQuotaExceededError,
    _redact_budget_identifiers,
    budget_exhaustion_detail,
    is_budget_exhausted_error,
    raise_if_budget_exhausted,
)
from cognee.infrastructure.llm.retry_config import (
    is_quota_or_billing_error,
    raise_if_quota_error,
    should_retry_llm_exception,
)

# The exact wordings litellm raises ``BudgetExceededError`` with. Sourced from
# litellm/exceptions.py, litellm/proxy/auth/auth_checks.py and
# litellm/proxy/hooks/model_max_budget_limiter.py. There are three sentence
# shapes, and in two of them a variable scope segment sits mid-sentence, so no
# single contiguous substring spans them all.
#
# Keep this list in sync with litellm on upgrade: a wording added upstream and
# missed here silently resurrects the retry storm for that budget scope.
LITELLM_BUDGET_MESSAGES = [
    # Shape 1 -- virtual key / team member / team / project / organization / tag
    "Budget has been exceeded! Current cost: 20.014285, Max budget: 20.0",
    "Budget has been exceeded! User=u1 in Team=t1 Current cost: 5.504, Max budget: 5.5",
    "Budget has been exceeded! Team=t1 Current cost: 70.004, Max budget: 70.0",
    "Budget has been exceeded! Project=p1 Current cost: 2.503, Max budget: 2.5",
    "Budget has been exceeded! Organization=o1 Current cost: 9.1, Max budget: 9.0",
    "Budget has been exceeded! Tag=prod Current cost: 1.2, Max budget: 1.0",
    # Shape 2 -- internal-user and end-user budgets (auth_checks.py:557,581,894)
    "ExceededBudget: User=u1 over budget. Spend=12.0, Budget=10.0",
    "ExceededBudget: End User=e1 over budget. Spend=3.0, Budget=2.0",
    # Shape 3 -- per-model budget caps (model_max_budget_limiter.py:80,135)
    "LiteLLM Virtual Key: tok, key_alias: ka, exceeded budget for model=gpt-4o",
    "LiteLLM End User: e1, exceeded budget for model=gpt-4o",
    # Observed verbatim from a real proxy (ghcr.io/berriai/litellm:main-stable,
    # a virtual key driven past its cap, HTTP 429) rather than read from
    # litellm's source. Appended rather than inserted: index 0 is the default
    # for ``_wrapped_budget_error`` and other tests key off its exact wording.
    # The key alias and key hint sit mid-sentence, which is the span the bounded
    # wildcard in ``_BUDGET_SENTENCE_RE`` has to cross.
    "Budget has been exceeded! Key=my-key-alias (sk-...-VGw) "
    "Current cost: 20.00066499999998, Max budget: 0.01",
]

# Prose that a cognified document could plausibly contain. None of it may be
# mistaken for a proxy rejection.
INNOCENT_PROSE = [
    "Q3 report: the marketing budget has been exceeded by 12%.",
    "If the budget has been exceeded, escalate to finance.",
    "Our max budget: 40000 EUR for the fiscal year.",
    "The travel budget has been exceeded again this quarter.",
    "The team is over budget. Spend=high, morale=low.",
    "We exceeded budget for the model rollout last year.",
    # Exclamation form, but no "Max budget:" label -- the conjunction is what
    # keeps this from matching.
    "The marketing budget has been exceeded! Please review before Friday.",
    "ExceededBudget was the title of the Q3 retrospective.",
]


def _rate_limit_error(message: str) -> litellm.RateLimitError:
    """The client-side error litellm raises for a proxy 429."""
    return litellm.RateLimitError(
        message=f"litellm.RateLimitError: RateLimitError: Litellm_proxyException - {message}",
        llm_provider="openai",
        model="litellm_proxy/litellm",
    )


def _wrapped_budget_error(
    message: str = LITELLM_BUDGET_MESSAGES[0],
    completion: str = "",
) -> InstructorRetryException:
    """Rebuild the exact chain prod produces: Instructor -> RetryError -> RateLimitError.

    ``completion`` populates the ``FailedAttempt`` so the model's own output is
    rendered into ``str()``, exactly as instructor does in production.
    """
    inner = _rate_limit_error(message)
    future = TenacityFuture(attempt_number=2)
    future.set_exception(inner)
    retry_error = RetryError(future)
    retry_error.__cause__ = inner

    exc = InstructorRetryException(
        str(inner),
        n_attempts=2,
        total_usage=0,
        failed_attempts=[
            FailedAttempt(attempt_number=1, exception=inner, completion=completion),
        ],
    )
    exc.__cause__ = retry_error
    return exc


class TestRealExceptionShape:
    """Pins the wrapper chain itself, so an instructor/litellm upgrade that
    changes it fails loudly here rather than silently in production."""

    def test_chain_is_what_we_think_it_is(self):
        exc = _wrapped_budget_error()
        chain = []
        current = exc
        while current is not None:
            chain.append(type(current).__name__)
            current = current.__cause__
        assert chain == ["InstructorRetryException", "RetryError", "RateLimitError"]

    def test_structured_signals_are_genuinely_unavailable(self):
        """Guards the rationale for text matching. If this ever fails, a
        structured check became possible and should be preferred."""
        exc = _wrapped_budget_error()
        assert getattr(exc, "status_code", None) is None
        assert not isinstance(exc, litellm.BudgetExceededError)

        deepest = exc.__cause__.__cause__
        assert deepest.status_code == 429
        # A different class from the server-side error, so isinstance cannot work.
        assert not isinstance(deepest, litellm.BudgetExceededError)

    def test_model_output_is_embedded_in_str(self):
        """The reason false positives are a live risk, not a theoretical one."""
        exc = _wrapped_budget_error(completion="the marketing budget has been exceeded by 12%")
        assert "marketing budget has been exceeded" in str(exc)

    def test_wrapped_error_is_classified_and_not_retried(self):
        exc = _wrapped_budget_error()
        assert is_budget_exhausted_error(exc) is True
        assert should_retry_llm_exception(exc) is False


class TestNoFalseNegatives:
    @pytest.mark.parametrize("message", LITELLM_BUDGET_MESSAGES)
    def test_every_litellm_budget_scope_is_terminal(self, message):
        exc = _wrapped_budget_error(message)
        assert is_budget_exhausted_error(exc) is True, f"not detected: {message}"
        assert should_retry_llm_exception(exc) is False, f"still retried: {message}"

    def test_bare_provider_error_is_terminal(self):
        """Unwrapped, as the non-instructor paths raise it."""
        exc = _rate_limit_error(LITELLM_BUDGET_MESSAGES[0])
        assert is_budget_exhausted_error(exc) is True
        assert should_retry_llm_exception(exc) is False

    def test_http_402_is_terminal_without_any_message_match(self):
        """Case 1 must keep working for direct-provider payment errors."""

        class PaymentRequired(Exception):
            status_code = 402

        assert is_budget_exhausted_error(PaymentRequired("payment required")) is True


class TestNoFalsePositives:
    @pytest.mark.parametrize("prose", INNOCENT_PROSE)
    def test_document_prose_is_not_budget_exhaustion(self, prose):
        exc = Exception(prose)
        assert is_budget_exhausted_error(exc) is False, f"false positive: {prose}"
        assert should_retry_llm_exception(exc) is True, f"wrongly terminal: {prose}"

    @pytest.mark.parametrize("prose", INNOCENT_PROSE)
    def test_prose_inside_a_failed_completion_is_not_budget_exhaustion(self, prose):
        """The dangerous case: a real validation-retry exhaustion whose partial
        completion quotes budget language must stay retryable."""
        inner = ValueError("validation failed: 3 attempts")
        exc = InstructorRetryException(
            str(inner),
            n_attempts=3,
            total_usage=0,
            failed_attempts=[
                FailedAttempt(attempt_number=1, exception=inner, completion=prose),
            ],
        )
        exc.__cause__ = inner
        assert is_budget_exhausted_error(exc) is False, f"false positive: {prose}"
        assert should_retry_llm_exception(exc) is True

    def test_fragments_far_apart_are_not_budget_exhaustion(self):
        """Detection matches a whole sentence, not fragments anywhere in the text.

        ``str(InstructorRetryException)`` concatenates every failed attempt's
        partial completion, so a document mentioning an exceeded budget in one
        paragraph and a maximum several KB later must not be classified.
        """
        doc = "the marketing budget has been exceeded! " + "x" * 4000 + " Our max budget: 40000 EUR"
        exc = Exception("validation failed, completion=" + doc)
        assert is_budget_exhausted_error(exc) is False
        assert should_retry_llm_exception(exc) is True

    def test_per_model_wording_requires_the_litellm_prefix(self):
        """Shape 3's distinguishing fragment is short enough to occur in prose,
        so the ``LiteLLM Virtual Key:`` / ``End User:`` prefix is required."""
        exc = Exception("the CI job exceeded budget for model=trained last sprint")
        assert is_budget_exhausted_error(exc) is False
        assert should_retry_llm_exception(exc) is True

    def test_detection_and_extraction_agree(self):
        """A positive classification must always yield a detail. Disagreement
        would mean the two used different rules again."""
        for message in LITELLM_BUDGET_MESSAGES:
            exc = _wrapped_budget_error(message)
            assert is_budget_exhausted_error(exc) is True
            assert budget_exhaustion_detail(exc) is not None

    def test_known_limitation_verbatim_provider_message_in_a_document(self):
        """Documents the one residual false positive, deliberately accepted.

        Text matching cannot distinguish a real rejection from a document that
        quotes one verbatim (a runbook, a pasted log, a GitHub issue). The
        trade-off is intentional and asymmetric: a false negative resurrects the
        production retry storm, whereas this costs one already-failing call a
        wrong 402 instead of a few more doomed retries. Pinned so the behaviour
        is a known quantity rather than a surprise.
        """
        quoted = (
            "Runbook: if the gateway logs "
            "'Budget has been exceeded! Current cost: 20.0, Max budget: 20.0', "
            "raise the tenant's cap."
        )
        assert is_budget_exhausted_error(Exception(quoted)) is True

    def test_ordinary_rate_limit_still_retries(self):
        """A real per-minute 429 has no budget wording and must stay retryable."""
        exc = _rate_limit_error("Rate limit reached for gpt-4o. Please try again in 20s.")
        assert is_budget_exhausted_error(exc) is False
        assert should_retry_llm_exception(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("read timeout"),
            ConnectionError("connection reset"),
            Exception("Internal server error"),
        ],
    )
    def test_transient_failures_still_retry(self, exc):
        assert is_budget_exhausted_error(exc) is False
        assert should_retry_llm_exception(exc) is True


class TestCauseChainWalk:
    def test_context_is_not_followed(self):
        """An unrelated error raised *while handling* a budget error must not be
        misclassified. ``__context__`` is set implicitly by nested raises;
        following it would produce false positives."""
        try:
            try:
                raise _rate_limit_error(LITELLM_BUDGET_MESSAGES[0])
            except Exception:
                raise ValueError("secondary failure during cleanup")
        except ValueError as e:
            unrelated = e

        assert unrelated.__context__ is not None  # the chain exists...
        assert unrelated.__cause__ is None  # ...but only via __context__
        assert is_budget_exhausted_error(unrelated) is False

    def test_cycle_in_cause_chain_terminates(self):
        a = Exception("a")
        b = Exception("b")
        a.__cause__ = b
        b.__cause__ = a
        assert is_budget_exhausted_error(a) is False  # must not hang

    def test_deep_chain_is_walked(self):
        exc = _rate_limit_error(LITELLM_BUDGET_MESSAGES[0])
        for i in range(50):
            outer = Exception(f"layer {i}")
            outer.__cause__ = exc
            exc = outer
        assert is_budget_exhausted_error(exc) is True


class TestErrorMapping:
    """Budget exhaustion is a 402, and must not be downgraded to the 422 quota
    error on its way through the gateway."""

    def test_payment_required_is_not_converted_to_quota_error(self):
        inner = _wrapped_budget_error()
        try:
            raise LLMPaymentRequiredError("LLM budget exhausted") from inner
        except LLMPaymentRequiredError as e:
            payment_error = e

        with pytest.raises(LLMPaymentRequiredError) as caught:
            raise_if_quota_error(payment_error)
        assert caught.value.status_code == 402

    def test_budget_wording_does_not_trip_the_quota_classifier(self):
        """``_TERMINAL_QUOTA_PATTERNS`` must stay free of budget wording, or
        ``raise_if_quota_error`` would rewrite the 402 into a 422."""
        assert is_quota_or_billing_error(_wrapped_budget_error()) is False

    def test_genuine_quota_errors_still_map_to_422(self):
        exc = Exception("You exceeded your current quota: insufficient_quota")
        assert is_quota_or_billing_error(exc) is True
        with pytest.raises(LLMQuotaExceededError):
            raise_if_quota_error(exc)


class TestDetailExtraction:
    def test_detail_is_the_provider_sentence_only(self):
        exc = _wrapped_budget_error(completion="x" * 5000)
        detail = budget_exhaustion_detail(exc)
        assert detail is not None
        assert detail.startswith("Budget has been exceeded!")
        assert detail.endswith("20.0")
        assert "xxxx" not in detail, "model output leaked into the user-facing detail"

    def test_detail_stays_short(self):
        """Bounded so a hostile completion cannot bloat the 402 response body."""
        exc = _wrapped_budget_error(completion="budget " * 2000)
        detail = budget_exhaustion_detail(exc)
        assert detail is not None and len(detail) < 300

    def test_detail_is_none_when_absent(self):
        assert budget_exhaustion_detail(Exception("something else")) is None

    def test_detail_is_bounded_against_an_unbroken_token(self):
        """The per-model shape ends in a bare token; without a length bound a
        100 KB model name would land verbatim in the 402 response body."""
        exc = Exception("LiteLLM Virtual Key: k, exceeded budget for model=" + "A" * 100_000)
        detail = budget_exhaustion_detail(exc)
        assert detail is None or len(detail) < 300

    def test_identifiers_are_redacted(self):
        """The sentence can carry the virtual-key hash, key alias and tenant
        IDs, and the detail is returned to API callers in the 402 body."""
        exc = Exception(
            "LiteLLM Virtual Key: sk-abc123secret, key_alias: prod-key, "
            "exceeded budget for model=gpt-4o"
        )
        detail = budget_exhaustion_detail(exc)
        assert detail is not None
        assert "sk-abc123secret" not in detail
        assert "prod-key" not in detail
        assert "<redacted>" in detail
        # The model is not an identifier and stays: it is the actionable part.
        assert "gpt-4o" in detail

    def test_scope_ids_are_redacted_but_figures_kept(self):
        detail = budget_exhaustion_detail(
            Exception(
                "Budget has been exceeded! Team=acme-corp Current cost: 70.0, Max budget: 70.0"
            )
        )
        assert detail is not None
        assert "acme-corp" not in detail
        assert "70.0" in detail

    def test_detail_walks_the_cause_chain(self):
        """Classification can succeed on a link below the outermost exception,
        so extraction has to walk too or the 402 loses its detail."""
        inner = Exception(LITELLM_BUDGET_MESSAGES[0])
        outer = Exception("wrapper with no budget wording")
        outer.__cause__ = inner
        assert is_budget_exhausted_error(outer) is True
        assert budget_exhaustion_detail(outer) is not None

    def test_detail_survives_a_raising_str(self):
        """``budget_exhaustion_detail`` runs inside ``raise_if_budget_exhausted``;
        a raising ``__str__`` must not replace the 402 with an unrelated error."""

        class Hostile(Exception):
            status_code = 402

            def __str__(self):
                raise RuntimeError("boom")

        exc = Hostile()
        assert is_budget_exhausted_error(exc) is True
        assert budget_exhaustion_detail(exc) is None
        with pytest.raises(LLMPaymentRequiredError):
            raise_if_budget_exhausted(exc)

    @pytest.mark.parametrize("message", LITELLM_BUDGET_MESSAGES)
    def test_detail_extracted_for_every_scope(self, message):
        detail = budget_exhaustion_detail(_wrapped_budget_error(message))
        assert detail is not None, f"no detail extracted for: {message}"
        assert len(detail) < 300
        # The detail is the provider sentence with identifiers masked -- nothing
        # invented, nothing pulled in from the surrounding wrapper noise.
        assert detail.lower() in _redact_budget_identifiers(message).lower()
