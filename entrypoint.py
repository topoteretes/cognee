#!/usr/bin/env python3
"""
Entry point for Cognee Docker container.
More portable than shell script for Windows compatibility.
"""
import os
import sys
import subprocess
import time

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result

def main():
    debug = os.getenv("DEBUG", "false")
    environment = os.getenv("ENVIRONMENT", "local")
    debug_port = os.getenv("DEBUG_PORT", "5678")
    http_port = os.getenv("HTTP_PORT", "8000")
    
    print(f"Debug mode: {debug}")
    print(f"Environment: {environment}")
    print(f"Debug port: {debug_port}")
    print(f"HTTP port: {http_port}")
    
    # Run Alembic migrations
    print("Running database migrations...")
    
    try:
        # Try running alembic migrations from cognee directory
        result = run_command(
            ["alembic", "upgrade", "head"],
            cwd="/app/cognee",
            check=False
        )
        if result.returncode != 0:
            raise Exception("Migration failed")
        print("Database migrations done.")
    except Exception as e:
        print(f"Migration failed: {e}")
        print("Trying to initialize database tables...")
        
        try:
            result = run_command(
                ["python", "/app/cognee/modules/engine/operations/setup.py"],
                check=False
            )
            if result.returncode != 0:
                raise Exception("Database initialization failed!")
        except Exception as init_error:
            print(f"Database initialization failed: {init_error}")
            sys.exit(1)
    
    print("Starting server...")
    
    # Add startup delay to ensure DB is ready
    time.sleep(2)
    
    # Build gunicorn command
    if environment in ("dev", "local"):
        if debug == "true":
            print("Starting in debug mode...")
            cmd = [
                "debugpy", "--wait-for-client", f"--listen=0.0.0.0:{debug_port}",
                "-m", "gunicorn",
                "-w", "1",
                "-k", "uvicorn.workers.UvicornWorker",
                "-t", "30000",
                f"--bind=0.0.0.0:{http_port}",
                "--log-level", "debug",
                "--reload",
                "--access-logfile", "-",
                "--error-logfile", "-",
                "cognee.api.client:app"
            ]
        else:
            cmd = [
                "gunicorn",
                "-w", "1",
                "-k", "uvicorn.workers.UvicornWorker",
                "-t", "30000",
                f"--bind=0.0.0.0:{http_port}",
                "--log-level", "debug",
                "--reload",
                "--access-logfile", "-",
                "--error-logfile", "-",
                "cognee.api.client:app"
            ]
    else:
        cmd = [
            "gunicorn",
            "-w", "1",
            "-k", "uvicorn.workers.UvicornWorker",
            "-t", "30000",
            f"--bind=0.0.0.0:{http_port}",
            "--log-level", "error",
            "--access-logfile", "-",
            "--error-logfile", "-",
            "cognee.api.client:app"
        ]
    
    print(f"Executing: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()
