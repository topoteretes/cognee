"""Optional x402 payment gating for cognee-mcp tool handlers.

When the ``imperial-a2a`` package is installed and ``COGNEE_X402_ENABLED=true``,
the ``@x402_gate`` decorator returns an HTTP 402 challenge for tool calls
that lack a valid ``X-402-Receipt`` header.  Paper mode (the default) simulates
the entire challenge/response flow without moving real money.

If ``imperial-a2a`` is **not** installed the decorator is a transparent no-op,
so existing cognee-mcp functionality is never affected.

Environment variables
---------------------
COGNEE_X402_ENABLED : str
    Set to ``"true"`` to activate payment gating.  Disabled by default.
X402_PAPER_MODE : str
    Set to ``"true"`` (the default) to simulate payments.  Set to ``"false"``
    only when you are ready to accept real on-chain payments.
"""

from __future__ import annotations

import functools
import json
import os
import uuid
from enum import Enum
from typing import Any, Callable

import mcp.types as types

# ---------------------------------------------------------------------------
# Payment tier definitions
# ---------------------------------------------------------------------------

class PaymentTier(str, Enum):
    """Price tiers available for gated tools."""

    MICRO = "micro"            # $0.01 - $0.10
    STANDARD = "standard"      # $0.10 - $1.00
    PREMIUM = "premium"        # $1.00 - $5.00
    ENTERPRISE = "enterprise"  # $5.00+

_TIER_PRICES: dict[PaymentTier, dict[str, Any]] = {
    PaymentTier.MICRO: {"amount": "0.01", "currency": "USD"},
    PaymentTier.STANDARD: {"amount": "0.25", "currency": "USD"},
    PaymentTier.PREMIUM: {"amount": "2.00", "currency": "USD"},
    PaymentTier.ENTERPRISE: {"amount": "10.00", "currency": "USD"},
}

# ---------------------------------------------------------------------------
# imperial-a2a availability probe
# ---------------------------------------------------------------------------

_HAS_IMPERIAL: bool
try:
    from imperial_a2a.x402 import verify_receipt as _verify_receipt  # type: ignore[import-untyped]
    _HAS_IMPERIAL = True
except ImportError:
    _HAS_IMPERIAL = False

    def _verify_receipt(*_args: Any, **_kwargs: Any) -> bool:  # noqa: D401
        """Stub that always returns True when imperial-a2a is absent."""
        return True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_x402_enabled() -> bool:
    return os.getenv("COGNEE_X402_ENABLED", "false").lower() == "true"

def _is_paper_mode() -> bool:
    return os.getenv("X402_PAPER_MODE", "true").lower() == "true"

def _build_402_challenge(tier: PaymentTier) -> list[types.TextContent]:
    """Return an MCP text response containing the 402 payment challenge."""
    pricing = _TIER_PRICES[tier]
    challenge = {
        "status": 402,
        "message": "Payment Required",
        "x402": {
            "version": "1.0",
            "tier": tier.value,
            "amount": pricing["amount"],
            "currency": pricing["currency"],
            "pay_to": os.getenv(
                "X402_PAY_TO",
                "0x0000000000000000000000000000000000000000",
            ),
            "network": os.getenv("X402_NETWORK", "base-sepolia"),
            "paper_mode": _is_paper_mode(),
            "challenge_id": str(uuid.uuid4()),
        },
    }
    return [types.TextContent(type="text", text=json.dumps(challenge))]

def _validate_receipt(receipt: str | None, tier: PaymentTier) -> bool:
    """Return True when the receipt is acceptable for *tier*."""
    if not receipt:
        return False

    # Paper mode: any non-empty receipt passes.
    if _is_paper_mode():
        return True

    # Live mode with imperial-a2a installed: delegate to library.
    if _HAS_IMPERIAL:
        pricing = _TIER_PRICES[tier]
        return _verify_receipt(
            receipt,
            min_amount=pricing["amount"],
            currency=pricing["currency"],
        )

    # Live mode but no library -- reject.
    return False

# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------

def x402_gate(tier: PaymentTier = PaymentTier.MICRO) -> Callable:
    """Decorator that gates an MCP tool behind x402 payment verification.

    Usage::

        @mcp.tool()
        @x402_gate(tier=PaymentTier.MICRO)
        async def my_tool(data: str) -> list:
            ...

    The decorator inspects the ``_x402_receipt`` keyword argument injected by
    the transport layer (or passed directly in tests).  When gating is
    disabled (the default) or ``imperial-a2a`` is not installed, the
    decorator is a transparent pass-through.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Fast path: gating not enabled -- run the tool unchanged.
            if not _is_x402_enabled():
                # Strip internal kwarg so the wrapped function is unaware.
                kwargs.pop("_x402_receipt", None)
                return await fn(*args, **kwargs)

            receipt = kwargs.pop("_x402_receipt", None)

            if not _validate_receipt(receipt, tier):
                return _build_402_challenge(tier)

            return await fn(*args, **kwargs)

        return wrapper

    return decorator
