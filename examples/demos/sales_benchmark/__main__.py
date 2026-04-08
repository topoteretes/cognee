"""Allow running with: python -m examples.demos.simple_agent_v2.sales_benchmark"""

from __future__ import annotations

import asyncio

from .run_benchmark import main

if __name__ == "__main__":
    asyncio.run(main())
