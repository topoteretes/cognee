"""Runtime compatibility for the cognee release installed inside the sandbox.

cognee's remote client (`serve`/`push`) opened its aiohttp session without
`trust_env=True`, so it ignored HTTP(S)_PROXY. Inside a Docker Sandbox that
means the request goes around the credential proxy and the placeholder API
key is never substituted (the brain answers 401). Fix upstream:
https://github.com/topoteretes/cognee/pull/5196. Until the kit installs a
release that carries it, make every aiohttp session honour the proxy env.

Detected by reading the installed client, not by version number: a release
cut before the fix lands (1.6.1 was) must still get the patch.
"""

from __future__ import annotations

import inspect

import aiohttp


def _client_ignores_proxy_env() -> bool:
    try:
        from cognee.api.v1.serve.cloud_client import CloudClient

        return "trust_env" not in inspect.getsource(CloudClient._get_session)
    except (ImportError, AttributeError, OSError, TypeError):
        return True  # cannot inspect the client: patching is harmless, not patching is a 401


if _client_ignores_proxy_env():
    _original_init = aiohttp.ClientSession.__init__

    def _init_with_trust_env(self, *args, **kwargs):
        kwargs.setdefault("trust_env", True)
        _original_init(self, *args, **kwargs)

    aiohttp.ClientSession.__init__ = _init_with_trust_env  # type: ignore[method-assign]
