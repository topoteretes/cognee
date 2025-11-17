import asyncio
import sys

async def run_in_local_sandbox(code: str) -> str:
    """
    Executes Python code in a separate process and captures its output.

    This function takes a string of Python code, executes it in an isolated
    subprocess using asyncio, and captures its standard output and standard error.

    Parameters:
    - code (str): A string containing the Python code to be executed.

    Returns:
    - str: The captured standard output from the executed code, or an error
           message if the execution fails.
    """
    process = None
    try:
        # Create a subprocess to run the Python code
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for the subprocess to finish, with a timeout
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)

        if process.returncode == 0:
            return stdout.decode()
        else:
            # If there was an error, return the decoded stderr
            return f"An error occurred:\n{stderr.decode()}"

    except asyncio.TimeoutError:
        # Ensure the process is terminated if it times out
        if process and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                # Process might have finished just before kill, which is fine.
                pass
        return "An error occurred: Code execution timed out after 60 seconds."
    except Exception as e:
        # Capture other potential exceptions
        return f"An error occurred: {e}"
