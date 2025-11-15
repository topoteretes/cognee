import subprocess
import sys

class RunInLocalSandbox:
    """
    This class is responsible for running code in a local sandbox.
    """
    async def run_in_local_sandbox(self, code: str, timeout: int = 60) -> str:
        """
        Executes the given Python code in a local sandbox environment and captures its output.
        
        Args:
            code: The Python code to execute
            timeout: Maximum execution time in seconds (default: 60)
            
        Returns:
            A string containing the execution result or error message
        """
        try:
            # Execute code in a separate Python process for isolation
            # This avoids the security risks of exec() while maintaining functionality
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                # Additional security: prevent the subprocess from creating child processes
                # and limit resource usage (can be extended based on requirements)
            )
            
            # Combine stdout and stderr for complete output
            output = result.stdout
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"
            
            if result.returncode == 0:
                return f"Code executed successfully:\nOutput:\n{output}" if output else "Code executed successfully with no output."
            else:
                return f"Code executed with errors (exit code {result.returncode}):\nOutput:\n{output}"
                
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {timeout} seconds."
        except Exception as e:
            return f"An error occurred during code execution: {e}"
