import time
import requests
from typing import Optional

def wait_for_health(url: str, timeout: int = 30, interval: int = 2) -> bool:
    """
    Poll health endpoint until ready or timeout.
    Mirrors logic from scripts/validate_docker_pull.sh
    """
    start = time.time()
    
    while start + timeout > time.time():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ Service healthy at {url}")
                return True
        except (requests.RequestException, requests.ConnectionError):
            pass
        
        time.sleep(interval)
    
    raise TimeoutError(f"Service failed to become healthy at {url} within {timeout}s")
