#!/usr/bin/env python3
"""
Cognee MCP (Management Control Plane) Client

A Python client for interacting with the Cognee MCP server API, allowing
for management and monitoring of Cognee infrastructure.
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_client")

class MCPError(Exception):
    """Base exception for MCP Client errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Any = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)
        
    def __str__(self):
        if self.status_code:
            return f"{self.message} (Status code: {self.status_code})"
        return self.message

class MCPClient:
    """Client for interacting with the Cognee MCP (Management Control Plane) server."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8080", 
        api_key: Optional[str] = None,
        timeout: int = 30,
        debug: bool = False
    ):
        """
        Initialize the MCP client.
        
        Args:
            base_url: Base URL of the MCP server
            api_key: API key for authentication
            timeout: Request timeout in seconds
            debug: Enable debug logging
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
    
    def _handle_response(self, response: requests.Response) -> Any:
        """Handle API response and error cases."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e}"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg = f"API error: {error_data['message']}"
            except:
                if response.text:
                    error_msg = f"API error: {response.text}"
            
            raise MCPError(error_msg, status_code=response.status_code, response_data=response.text)
        except requests.exceptions.JSONDecodeError:
            if response.text:
                return {"response": response.text}
            return {}
        except Exception as e:
            raise MCPError(f"Unexpected error: {str(e)}")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make an HTTP request to the MCP API."""
        url = urljoin(self.base_url, endpoint)
        headers = self._get_headers()
        
        logger.debug(f"Making {method} request to {url}")
        if params:
            logger.debug(f"Request params: {params}")
        if data:
            logger.debug(f"Request data: {json.dumps(data)}")
            
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=self.timeout
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise MCPError(f"Request failed: {e}")
    
    # Server status and health endpoints
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the MCP server."""
        return self._make_request("GET", "/api/v1/status")
    
    def get_health(self) -> Dict[str, Any]:
        """Perform a health check on the MCP server."""
        return self._make_request("GET", "/api/v1/health")
    
    # Configuration management
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of the MCP server."""
        return self._make_request("GET", "/api/v1/config")
    
    def update_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the MCP server configuration.
        
        Args:
            config_data: Configuration data to update
            
        Returns:
            The updated configuration
        """
        return self._make_request("PATCH", "/api/v1/config", data=config_data)
    
    # Service management
    
    def get_services(self) -> List[Dict[str, Any]]:
        """Get a list of services managed by the MCP server."""
        response = self._make_request("GET", "/api/v1/services")
        return response.get("services", [])
    
    def get_service(self, service_id: str) -> Dict[str, Any]:
        """
        Get details for a specific service.
        
        Args:
            service_id: ID of the service
            
        Returns:
            Service details
        """
        return self._make_request("GET", f"/api/v1/services/{service_id}")
    
    def start_service(self, service_id: str) -> Dict[str, Any]:
        """
        Start a service.
        
        Args:
            service_id: ID of the service to start
            
        Returns:
            Service status
        """
        return self._make_request("POST", f"/api/v1/services/{service_id}/start")
    
    def stop_service(self, service_id: str) -> Dict[str, Any]:
        """
        Stop a service.
        
        Args:
            service_id: ID of the service to stop
            
        Returns:
            Service status
        """
        return self._make_request("POST", f"/api/v1/services/{service_id}/stop")
    
    def restart_service(self, service_id: str) -> Dict[str, Any]:
        """
        Restart a service.
        
        Args:
            service_id: ID of the service to restart
            
        Returns:
            Service status
        """
        return self._make_request("POST", f"/api/v1/services/{service_id}/restart")
    
    # Resource management
    
    def get_resources(self, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of resources managed by the MCP server.
        
        Args:
            resource_type: Filter by resource type
            
        Returns:
            List of resources
        """
        params = {}
        if resource_type:
            params["type"] = resource_type
            
        response = self._make_request("GET", "/api/v1/resources", params=params)
        return response.get("resources", [])
    
    def get_resource(self, resource_id: str) -> Dict[str, Any]:
        """
        Get details for a specific resource.
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            Resource details
        """
        return self._make_request("GET", f"/api/v1/resources/{resource_id}")
    
    def create_resource(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new resource.
        
        Args:
            resource_data: Resource configuration
            
        Returns:
            Created resource details
        """
        return self._make_request("POST", "/api/v1/resources", data=resource_data)
    
    def update_resource(self, resource_id: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a resource.
        
        Args:
            resource_id: ID of the resource to update
            resource_data: Resource configuration
            
        Returns:
            Updated resource details
        """
        return self._make_request("PATCH", f"/api/v1/resources/{resource_id}", data=resource_data)
    
    def delete_resource(self, resource_id: str) -> Dict[str, Any]:
        """
        Delete a resource.
        
        Args:
            resource_id: ID of the resource to delete
            
        Returns:
            Deletion status
        """
        return self._make_request("DELETE", f"/api/v1/resources/{resource_id}")
    
    # User management
    
    def get_users(self) -> List[Dict[str, Any]]:
        """Get a list of users."""
        response = self._make_request("GET", "/api/v1/users")
        return response.get("users", [])
    
    def get_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get details for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            User details
        """
        return self._make_request("GET", f"/api/v1/users/{user_id}")
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            user_data: User information
            
        Returns:
            Created user details
        """
        return self._make_request("POST", "/api/v1/users", data=user_data)
    
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a user.
        
        Args:
            user_id: ID of the user to update
            user_data: User information
            
        Returns:
            Updated user details
        """
        return self._make_request("PATCH", f"/api/v1/users/{user_id}", data=user_data)
    
    def delete_user(self, user_id: str) -> Dict[str, Any]:
        """
        Delete a user.
        
        Args:
            user_id: ID of the user to delete
            
        Returns:
            Deletion status
        """
        return self._make_request("DELETE", f"/api/v1/users/{user_id}")
    
    # Docker image management
    
    def get_images(self) -> List[Dict[str, Any]]:
        """Get a list of available Docker images."""
        response = self._make_request("GET", "/api/v1/images")
        return response.get("images", [])
    
    def pull_image(self, image_name: str, image_tag: str = "latest") -> Dict[str, Any]:
        """
        Pull a Docker image.
        
        Args:
            image_name: Name of the Docker image
            image_tag: Tag of the Docker image
            
        Returns:
            Pull status
        """
        data = {
            "name": image_name,
            "tag": image_tag
        }
        return self._make_request("POST", "/api/v1/images/pull", data=data)
    
    def delete_image(self, image_id: str) -> Dict[str, Any]:
        """
        Delete a Docker image.
        
        Args:
            image_id: ID of the Docker image to delete
            
        Returns:
            Deletion status
        """
        return self._make_request("DELETE", f"/api/v1/images/{image_id}")
    
    # Logs and metrics
    
    def get_logs(self, service_id: Optional[str] = None, lines: int = 100) -> Dict[str, Any]:
        """
        Get system or service logs.
        
        Args:
            service_id: ID of the service (optional)
            lines: Number of log lines to retrieve
            
        Returns:
            Log entries
        """
        params = {"lines": lines}
        if service_id:
            return self._make_request("GET", f"/api/v1/services/{service_id}/logs", params=params)
        return self._make_request("GET", "/api/v1/logs", params=params)
    
    def get_metrics(
        self, 
        service_id: Optional[str] = None, 
        duration: str = "5m"
    ) -> Dict[str, Any]:
        """
        Get system or service metrics.
        
        Args:
            service_id: ID of the service (optional)
            duration: Duration for metrics (e.g., "5m", "1h")
            
        Returns:
            Metrics data
        """
        params = {"duration": duration}
        if service_id:
            return self._make_request("GET", f"/api/v1/services/{service_id}/metrics", params=params)
        return self._make_request("GET", "/api/v1/metrics", params=params)


def test_mcp_client():
    """Simple test function to demonstrate usage of the MCPClient."""
    # Create client
    client = MCPClient(
        base_url=os.environ.get("COGNEE_MCP_URL", "http://localhost:8080"),
        api_key=os.environ.get("COGNEE_MCP_API_KEY"),
        debug=True
    )
    
    try:
        # Test server status
        print("Getting server status...")
        status = client.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Test server health
        print("\nGetting server health...")
        health = client.get_health()
        print(f"Health: {json.dumps(health, indent=2)}")
        
        # Test configuration
        print("\nGetting server configuration...")
        config = client.get_config()
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Test services
        print("\nGetting list of services...")
        services = client.get_services()
        print(f"Found {len(services)} services")
        
        if services:
            service_id = services[0].get("id", "")
            if service_id:
                print(f"\nGetting details for service {service_id}...")
                service_details = client.get_service(service_id)
                print(f"Service details: {json.dumps(service_details, indent=2)}")
        
    except MCPError as e:
        print(f"Error: {e}")
        if e.response_data:
            print(f"Response data: {e.response_data}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_mcp_client()