import io
import sys
import json
import traceback


def wrap_in_async_handler(user_code: str) -> str:
    return (
        "import asyncio\n\n"
        "from cognee.infrastructure.utils.run_sync import run_sync\n\n"
        "async def __user_main__():\n"
        + "\n".join("    " + line for line in user_code.strip().split("\n"))
        + "\n"
        "    globals().update(locals())\n\n"
        "run_sync(__user_main__())\n"
    )


def run_in_local_sandbox(code, environment=None):
    environment = environment or {}
    code = wrap_in_async_handler(code.replace("\xa0", "\n"))

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    sys.stderr = buffer

    error = None

    try:
        exec(code, environment)
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stdout

    output = buffer.getvalue()

    if output:
        if "\n" in output:
            output_items = output.split("\n")

            def process_output(output):
                try:
                    result = json.loads(output)
                    return result
                except json.JSONDecodeError:
                    return output

            output = list(map(process_output, output_items))

    return output, error


if __name__ == "__main__":
    run_in_local_sandbox("""
import cognee

await cognee.add("Test file with some random content 3.")

a = "asd"

b = {"c": "dfgh"}
""")
