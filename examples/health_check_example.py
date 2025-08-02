#!/usr/bin/env python3
"""Example script showing how to use the health check endpoints."""

import requests
import json
import sys


def test_health_endpoints(base_url="http://localhost:8000"):
    """Test all health check endpoints."""
    
    print(f"Testing health endpoints at {base_url}")
    print("=" * 50)
    
    # Test basic health endpoint
    print("\n1. Testing basic health endpoint (/health)")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text if response.text else 'Empty response'}")
    except requests.RequestException as e:
        print(f"Error: {e}")
    
    # Test readiness endpoint
    print("\n2. Testing readiness endpoint (/health/ready)")
    try:
        response = requests.get(f"{base_url}/health/ready", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.headers.get('content-type', '').startswith('application/json'):
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error: {e}")
    
    # Test detailed health endpoint
    print("\n3. Testing detailed health endpoint (/health/detailed)")
    try:
        response = requests.get(f"{base_url}/health/detailed", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.headers.get('content-type', '').startswith('application/json'):
            health_data = response.json()
            print(f"Overall Status: {health_data.get('status', 'unknown')}")
            print(f"Version: {health_data.get('version', 'unknown')}")
            print(f"Uptime: {health_data.get('uptime', 0)} seconds")
            print("\nComponent Status:")
            for component, details in health_data.get('components', {}).items():
                print(f"  {component}: {details.get('status')} ({details.get('provider')}) - {details.get('response_time_ms')}ms")
                if details.get('details'):
                    print(f"    Details: {details.get('details')}")
        else:
            print(f"Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error: {e}")


def monitor_health(base_url="http://localhost:8000", interval=30):
    """Continuously monitor health status."""
    import time
    
    print(f"Monitoring health at {base_url} every {interval} seconds")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            try:
                response = requests.get(f"{base_url}/health/detailed", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    timestamp = data.get('timestamp', 'unknown')
                    print(f"[{timestamp}] Status: {status}")
                    
                    # Show any unhealthy components
                    unhealthy = [
                        name for name, comp in data.get('components', {}).items()
                        if comp.get('status') != 'healthy'
                    ]
                    if unhealthy:
                        print(f"  Issues: {', '.join(unhealthy)}")
                else:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HTTP {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Connection error: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
            monitor_health(base_url)
        else:
            test_health_endpoints(sys.argv[1])
    else:
        test_health_endpoints()
        
    print("\nUsage:")
    print("  python health_check_example.py                    # Test endpoints")
    print("  python health_check_example.py http://host:port   # Test specific host")
    print("  python health_check_example.py monitor            # Monitor continuously")