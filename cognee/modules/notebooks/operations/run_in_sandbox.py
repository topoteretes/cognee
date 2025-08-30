import json
import modal


def wrapper_in_cognee_code(user_code):
    return f"""
import json
import asyncio
from fastapi.encoders import jsonable_encoder
from cognee import search, SearchType

async def main():
    async def user_invocation():
        {user_code}
  
    results = await search("What is in the data?", SearchType.GRAPH_COMPLETION)

    jsonable_results = jsonable_encoder(results)
    print(json.dumps(jsonable_results))

asyncio.run(main())
"""


def run_in_sandbox(code):
    code = code.replace("\xa0", "")

    from cognee.root_dir import get_absolute_path

    sandbox_image = modal.Image.debian_slim().uv_pip_install("cognee==0.2.3.dev1")
    secrets = modal.Secret.from_dotenv(path=get_absolute_path("../.env"))

    sandbox_app = modal.App.lookup("code-sandbox-app", create_if_missing=True)

    sandbox = modal.Sandbox.create(
        app=sandbox_app,
        name="code-sandbox",
        image=sandbox_image,
        secrets=[secrets],
        timeout=600,
    )
    sandbox_id = sandbox.object_id

    code_sandbox = modal.Sandbox.from_id(sandbox_id)

    try:
        process = code_sandbox.exec("python", "-c", code)
        output = process.stdout.read()

        if output:
            output_items = output.split("\n")

            def process_output(output):
                try:
                    result = json.loads(output)
                    return result
                except json.JSONDecodeError:
                    return output

            output = list(map(process_output, output_items))

        try:
            error = process.stderr.read()
        except Exception as e:
            error = str(e)
    finally:
        code_sandbox.terminate()

    return output, error


if __name__ == "__main__":
    run_in_sandbox(wrapper_in_cognee_code(""))
