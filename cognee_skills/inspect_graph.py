import asyncio
from pathlib import Path
import cognee

ROOT = Path(__file__).resolve().parent
COGNEE_SYSTEM = ROOT / ".cognee_system"


async def main():
    cognee.config.system_root_directory(str(COGNEE_SYSTEM))
    out = ROOT / "graph.html"
    await cognee.visualize_graph(str(out))
    print(f"Open in browser: file://{out}")


if __name__ == "__main__":
    asyncio.run(main())
