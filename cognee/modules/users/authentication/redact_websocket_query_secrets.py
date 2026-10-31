"""Keep the WebSocket query-param auth fallback's credential out of uvicorn's
own logs.

``resolve_websocket_query_param_fallback`` accepts a bearer/API-key token via
``?token=`` because browsers cannot set custom headers on a WebSocket
handshake. That credential is not only exposed to whatever a deployment's
reverse proxy logs (operators are told to redact it there) — uvicorn itself
logs the full handshake path, query string included, on every accept/close at
INFO level via the ``uvicorn.error`` logger (see
``uvicorn.protocols.websockets.websockets_sansio_impl.WebSocketsSansIOProtocol
.asgi_send``), regardless of any proxy in front of it. That sink is entirely
within this codebase's control, unlike a proxy's own configuration, so it is
redacted here rather than merely documented.
"""

import logging
import re

from .websocket_query_param import WEBSOCKET_QUERY_PARAM_NAME

_QUERY_PARAM_RE = re.compile(
    rf"([?&]){re.escape(WEBSOCKET_QUERY_PARAM_NAME)}=[^&\s\"']+", re.IGNORECASE
)


class _RedactWebsocketTokenFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _QUERY_PARAM_RE.sub(
                rf"\1{WEBSOCKET_QUERY_PARAM_NAME}=***REDACTED***", record.msg
            )
        if record.args and not isinstance(record.args, dict):
            # `%(name)s`-style logging passes a dict here instead of a
            # positional tuple; rebuilding it as a tuple of its keys would
            # corrupt %-formatting downstream. Nothing in uvicorn's own
            # access/error loggers does that today, but leave dict args
            # untouched rather than assuming the format never changes.
            record.args = tuple(
                _QUERY_PARAM_RE.sub(rf"\1{WEBSOCKET_QUERY_PARAM_NAME}=***REDACTED***", arg)
                if isinstance(arg, str)
                else arg
                for arg in record.args
            )
        return True


_installed = False


def install_websocket_query_param_redaction() -> None:
    """Idempotent: safe to call every time the ASGI app module is imported."""
    global _installed
    if _installed:
        return
    _installed = True

    redaction_filter = _RedactWebsocketTokenFilter()
    # uvicorn.error carries the WebSocket accept/close/status lines (with the
    # full path+query string); uvicorn.access carries the plain HTTP request
    # log line uvicorn emits for the handshake's initial GET/Upgrade request.
    logging.getLogger("uvicorn.error").addFilter(redaction_filter)
    logging.getLogger("uvicorn.access").addFilter(redaction_filter)
