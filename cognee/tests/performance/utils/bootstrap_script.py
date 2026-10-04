# Importing locust at module top monkey-patches socket/time via gevent, which
# makes asyncio + async sqlite hot-spin at 100% CPU. Run bootstrap in a fresh
# interpreter that never imports locust.

import asyncio
import os
import sys

import cognee
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.api_key.create_api_key import create_api_key
from cognee.modules.users.methods import create_default_user


async def main(out_path: str) -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()
    user = await create_default_user()
    api_key_obj = await create_api_key(user, name="locust-loadtest")
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # os.open only applies the mode when it creates the file; enforce 0o600 for
    # a pre-existing file too so the secret is never left world-readable.
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(api_key_obj.api_key)


asyncio.run(main(sys.argv[1]))
