# Importing locust at module top monkey-patches socket/time via gevent, which
# makes asyncio + async sqlite hot-spin at 100% CPU. Run bootstrap in a fresh
# interpreter that never imports locust.

import asyncio
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
    with open(out_path, "w") as f:
        f.write(api_key_obj.api_key)


asyncio.run(main(sys.argv[1]))
