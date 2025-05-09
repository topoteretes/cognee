#!/usr/bin/env python3
"""
Script to run the Cognee API server for testing.
"""

import os
import sys
import argparse
from cognee.api.client import start_api_server

def main():
    """Run the Cognee API server with specified host and port."""
    parser = argparse.ArgumentParser(description="Run the Cognee API server for testing.")
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--env", 
        choices=["prod", "dev", "local"], 
        default="local",
        help="Environment to run the server in (default: local)"
    )
    
    args = parser.parse_args()
    
    # Set environment variable
    os.environ["ENV"] = args.env
    
    print(f"Starting Cognee API server in {args.env} mode on {args.host}:{args.port}")
    
    try:
        start_api_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 