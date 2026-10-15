"""Minimization plus encryption, not a claim of anonymization."""

import hashlib
import hmac
import json
import os
import re

from cryptography.fernet import Fernet, MultiFernet

POLICY_VERSION = "answer-evaluation-v1"
_PATTERNS = [
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
    r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+",
    r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[^\s,;]+",
    r"\b(?:sk-(?:proj-|ant-)?|gh[pousr]_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{12,}",
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    r"(?i)\b(?:https?|postgres(?:ql)?|redis|mongodb)://[^\s<>]+",
    r"(?:/Users/|/home/|[A-Z]:\\Users\\)[^\s/\\]+",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
]


def sanitize(value):
    mapping = {}

    def text(s):
        def replace(match):
            original = match.group(0)
            if original not in mapping:
                mapping[original] = f"[REDACTED_{len(mapping) + 1}]"
            return mapping[original]

        for pattern in _PATTERNS:
            s = re.sub(pattern, replace, s)
        return s

    def visit(v):
        if isinstance(v, str):
            return text(v)
        if isinstance(v, list):
            return [visit(x) for x in v]
        if isinstance(v, dict):
            return {k: visit(x) for k, x in v.items()}
        return v

    return visit(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class Protection:
    def __init__(self):
        keys = os.environ.get("COGNEE_EVALUATION_ENCRYPTION_KEYS", "").split(",")
        self.cipher = MultiFernet([Fernet(k.strip().encode()) for k in keys if k.strip()])
        self.secret = os.environ.get("COGNEE_EVALUATION_ID_KEY", "").encode()
        if len(self.secret) < 32:
            raise ValueError("Evaluation identity key must contain at least 32 bytes")

    def identity(self, *values):
        return hmac.new(self.secret, canonical(values).encode(), hashlib.sha256).hexdigest()

    def encrypt(self, value):
        return self.cipher.encrypt(canonical(value).encode()).decode()

    def decrypt(self, value):
        return json.loads(self.cipher.decrypt(value.encode()))
