import io
import sys
import traceback


def wrap_in_async_handler(user_code: str) -> str:
    return (
        "import asyncio\n"
        + "asyncio.set_event_loop(running_loop)\n\n"
        + "from cognee.infrastructure.utils.run_sync import run_sync\n\n"
        + "async def __user_main__():\n"
        + "\n".join("    " + line for line in user_code.strip().split("\n"))
        + "\n"
        + "    globals().update(locals())\n\n"
        + "run_sync(__user_main__(), running_loop)\n"
    )


def run_in_local_sandbox(code, environment=None, loop=None):
    environment = environment or {}
    code = wrap_in_async_handler(code.replace("\xa0", "\n"))

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    sys.stderr = buffer

    error = None

    printOutput = []

    def customPrintFunction(output):
        printOutput.append(output)

    environment["print"] = customPrintFunction
    environment["running_loop"] = loop

    try:
        exec(code, environment)
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stdout

    return printOutput, error


if __name__ == "__main__":
    run_in_local_sandbox("""
import cognee

await cognee.add("Test file with some random content 3.")

a = "asd"

b = {"c": "dfgh"}
""")
