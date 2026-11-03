"""``import cognee`` must not import litellm.

litellm (and the OpenAI SDK it imports) is half of cognee's import time and is only
needed on the first LLM or embedding call; the keyless path never calls it. The import
runs in a subprocess so the modules this test process already loaded do not count.
"""

import json
import os
import subprocess
import sys

# The OpenAI SDK is not asserted: ``cognee.infrastructure.llm`` installs its
# type-introspection cache into that SDK at import, which is a separate concern.
PROBE = """
import json, sys
import cognee
print(json.dumps({name: name in sys.modules for name in ("litellm", "instructor")}))
"""


def test_import_cognee_does_not_load_the_llm_libraries():
    output = subprocess.run(
        [sys.executable, "-c", PROBE],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "COGNEE_SKIP_CONNECTION_TEST": "true"},
    ).stdout
    loaded = json.loads(output.strip().splitlines()[-1])

    assert loaded == {"litellm": False, "instructor": False}
