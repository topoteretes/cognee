"""Tests for the optional x402 payment gating middleware."""

import json
import os
import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from src.x402_middleware import (
    PaymentTier,
    _build_402_challenge,
    _is_paper_mode,
    _is_x402_enabled,
    _validate_receipt,
    x402_gate,
)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def test_x402_disabled_by_default():
    os.environ.pop("COGNEE_X402_ENABLED", None)
    assert _is_x402_enabled() is False

def test_x402_enabled_via_env(monkeypatch):
    monkeypatch.setenv("COGNEE_X402_ENABLED", "true")
    assert _is_x402_enabled() is True

def test_paper_mode_on_by_default():
    os.environ.pop("X402_PAPER_MODE", None)
    assert _is_paper_mode() is True

def test_paper_mode_off(monkeypatch):
    monkeypatch.setenv("X402_PAPER_MODE", "false")
    assert _is_paper_mode() is False

# ---------------------------------------------------------------------------
# 402 challenge shape
# ---------------------------------------------------------------------------

def test_challenge_contains_required_fields():
    result = _build_402_challenge(PaymentTier.MICRO)
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["status"] == 402
    assert payload["message"] == "Payment Required"
    x402 = payload["x402"]
    assert x402["version"] == "1.0"
    assert x402["tier"] == "micro"
    assert x402["amount"] == "0.01"
    assert x402["currency"] == "USD"
    assert "challenge_id" in x402

def test_challenge_tiers_have_increasing_prices():
    tiers = [PaymentTier.MICRO, PaymentTier.STANDARD, PaymentTier.PREMIUM, PaymentTier.ENTERPRISE]
    prices = []
    for tier in tiers:
        result = _build_402_challenge(tier)
        payload = json.loads(result[0].text)
        prices.append(float(payload["x402"]["amount"]))
    assert prices == sorted(prices)
    assert len(set(prices)) == len(prices)  # all distinct

# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------

def test_empty_receipt_rejected():
    assert _validate_receipt(None, PaymentTier.MICRO) is False
    assert _validate_receipt("", PaymentTier.MICRO) is False

def test_paper_mode_accepts_any_receipt(monkeypatch):
    monkeypatch.setenv("X402_PAPER_MODE", "true")
    assert _validate_receipt("paper-receipt-abc", PaymentTier.PREMIUM) is True

# ---------------------------------------------------------------------------
# Decorator behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decorator_passthrough_when_disabled(monkeypatch):
    """When COGNEE_X402_ENABLED is not set, the tool runs normally."""
    monkeypatch.delenv("COGNEE_X402_ENABLED", raising=False)

    @x402_gate(tier=PaymentTier.MICRO)
    async def dummy_tool(data: str) -> str:
        return f"processed: {data}"

    result = await dummy_tool("hello")
    assert result == "processed: hello"

@pytest.mark.asyncio
async def test_decorator_returns_402_when_no_receipt(monkeypatch):
    """With gating on and no receipt, tool returns 402 challenge."""
    monkeypatch.setenv("COGNEE_X402_ENABLED", "true")
    monkeypatch.setenv("X402_PAPER_MODE", "true")

    @x402_gate(tier=PaymentTier.STANDARD)
    async def gated_tool(data: str) -> str:
        return f"processed: {data}"

    result = await gated_tool("hello")
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["status"] == 402
    assert payload["x402"]["tier"] == "standard"

@pytest.mark.asyncio
async def test_decorator_allows_with_valid_receipt(monkeypatch):
    """With gating on and a valid paper receipt, tool executes."""
    monkeypatch.setenv("COGNEE_X402_ENABLED", "true")
    monkeypatch.setenv("X402_PAPER_MODE", "true")

    @x402_gate(tier=PaymentTier.MICRO)
    async def gated_tool(data: str) -> str:
        return f"processed: {data}"

    result = await gated_tool("hello", _x402_receipt="paper-receipt-123")
    assert result == "processed: hello"

@pytest.mark.asyncio
async def test_decorator_strips_internal_kwarg_when_disabled(monkeypatch):
    """The _x402_receipt kwarg must not leak to the wrapped function."""
    monkeypatch.delenv("COGNEE_X402_ENABLED", raising=False)

    @x402_gate(tier=PaymentTier.MICRO)
    async def strict_tool(data: str) -> str:
        return f"processed: {data}"

    # Would raise TypeError if _x402_receipt leaked through.
    result = await strict_tool("hello", _x402_receipt="anything")
    assert result == "processed: hello"
