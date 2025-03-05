#!/usr/bin/env python3
"""
Docker Image Testing for Cognee MCP

This script focuses on testing the Docker image functionality of the Cognee MCP server,
including pulling, verifying, and running containers from the Docker image.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from mcp_client import MCPClient, MCPError


class DockerImageTester:
    """Test the Docker image functionality of the MCP server."""
    
    def __init__(
        self, 
        mcp_client: MCPClient,
        image_name: str = "cognee/cognee-mcp",
        image_tag: str = "latest",
        verbose: bool = False
    ):
        """
        Initialize the Docker image tester.
        
        Args:
            mcp_client: An instance of the MCPClient
            image_name: Name of the Docker image to test
            image_tag: Tag of the Docker image to test
            verbose: Enable verbose output
        """
        self.client = mcp_client
        self.image_name = image_name
        self.image_tag = image_tag
        self.full_image = f"{image_name}:{image_tag}"
        self.verbose = verbose
        
    def log(self, message: str) -> None:
        """Log a message."""
        print(f"[DockerTester] {message}")
        
    def run_command(self, command: List[str]) -> Tuple[int, str, str]:
        """
        Run a shell command and return the exit code, stdout, and stderr.
        
        Args:
            command: Command to run as a list of strings
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if self.verbose:
            self.log(f"Running command: {' '.join(command)}")
            
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
        if self.verbose:
            if stdout:
                self.log(f"Command stdout: {stdout}")
            if stderr:
                self.log(f"Command stderr: {stderr}")
                
        return exit_code, stdout, stderr
    
    def verify_docker_available(self) -> bool:
        """
        Verify that Docker is available on the system.
        
        Returns:
            True if Docker is available, False otherwise
        """
        self.log("Verifying Docker availability...")
        exit_code, stdout, stderr = self.run_command(["docker", "--version"])
        
        if exit_code != 0:
            self.log("Docker is not available. Please install Docker and try again.")
            return False
            
        self.log(f"Docker is available: {stdout.strip()}")
        return True
    
    def check_image_exists_locally(self) -> bool:
        """
        Check if the Docker image exists locally.
        
        Returns:
            True if the image exists locally, False otherwise
        """
        self.log(f"Checking if image {self.full_image} exists locally...")
        exit_code, stdout, stderr = self.run_command([
            "docker", "image", "inspect", self.full_image
        ])
        
        if exit_code != 0:
            self.log(f"Image {self.full_image} does not exist locally.")
            return False
            
        self.log(f"Image {self.full_image} exists locally.")
        return True
    
    def pull_image(self) -> bool:
        """
        Pull the Docker image from the registry.
        
        Returns:
            True if the image was pulled successfully, False otherwise
        """
        self.log(f"Pulling image {self.full_image}...")
        
        # First try using the MCP client
        try:
            result = self.client.pull_image(self.image_name, self.image_tag)
            self.log(f"Image pulled successfully via MCP client: {result}")
            return True
        except MCPError as e:
            self.log(f"Failed to pull image via MCP client: {e}")
            self.log("Trying to pull directly using Docker command...")
        
        # Fallback to using Docker command
        exit_code, stdout, stderr = self.run_command([
            "docker", "pull", self.full_image
        ])
        
        if exit_code != 0:
            self.log(f"Failed to pull image {self.full_image}: {stderr}")
            return False
            
        self.log(f"Image {self.full_image} pulled successfully.")
        return True
    
    def get_image_details(self) -> Optional[Dict[str, Any]]:
        """
        Get details about the Docker image.
        
        Returns:
            Image details as a dictionary, or None if the image doesn't exist
        """
        self.log(f"Getting details for image {self.full_image}...")
        exit_code, stdout, stderr = self.run_command([
            "docker", "image", "inspect", self.full_image
        ])
        
        if exit_code != 0:
            self.log(f"Failed to get details for image {self.full_image}: {stderr}")
            return None
            
        try:
            details = json.loads(stdout)[0]
            return details
        except (json.JSONDecodeError, IndexError) as e:
            self.log(f"Failed to parse image details: {e}")
            return None
    
    def verify_image(self) -> bool:
        """
        Verify the Docker image contents and configuration.
        
        Returns:
            True if the image verification passes, False otherwise
        """
        self.log(f"Verifying image {self.full_image}...")
        
        details = self.get_image_details()
        if not details:
            return False
        
        # Check for key metadata
        repo_tags = details.get("RepoTags", [])
        if not repo_tags or self.full_image not in repo_tags:
            self.log(f"Image tag verification failed. Expected {self.full_image}, got {repo_tags}")
            return False
        
        # Check image architecture
        architecture = details.get("Architecture")
        if not architecture:
            self.log("Image architecture not found")
            return False
        
        self.log(f"Image architecture: {architecture}")
        
        # Check for exposed ports
        config = details.get("Config", {})
        exposed_ports = config.get("ExposedPorts", {})
        
        if not exposed_ports:
            self.log("Warning: No exposed ports found in the image")
        else:
            self.log(f"Exposed ports: {', '.join(exposed_ports.keys())}")
        
        # Check for environment variables
        env_vars = config.get("Env", [])
        if not env_vars:
            self.log("Warning: No environment variables found in the image")
        else:
            self.log(f"Found {len(env_vars)} environment variables")
            
        # Check image size
        size_bytes = details.get("Size", 0)
        size_mb = size_bytes / (1024 * 1024)
        self.log(f"Image size: {size_mb:.2f} MB")
        
        # Check image creation date
        created = details.get("Created")
        if created:
            self.log(f"Image created: {created}")
            
        self.log("Image verification completed successfully")
        return True
    
    def run_container(self, container_name: str = "mcp-test", ports: Dict[str, str] = None) -> Optional[str]:
        """
        Run a container from the Docker image.
        
        Args:
            container_name: Name to give the container
            ports: Port mappings (host:container)
            
        Returns:
            Container ID if successful, None otherwise
        """
        if ports is None:
            ports = {"8080": "8080"}  # Default port mapping
            
        port_args = []
        for host_port, container_port in ports.items():
            port_args.extend(["-p", f"{host_port}:{container_port}"])
            
        self.log(f"Running container {container_name} from image {self.full_image}...")
        command = [
            "docker", "run", "--name", container_name, "-d",
            *port_args, self.full_image
        ]
        
        exit_code, stdout, stderr = self.run_command(command)
        
        if exit_code != 0:
            self.log(f"Failed to run container: {stderr}")
            return None
            
        container_id = stdout.strip()
        self.log(f"Container started with ID: {container_id}")
        return container_id
    
    def stop_container(self, container_id: str) -> bool:
        """
        Stop a running container.
        
        Args:
            container_id: ID of the container to stop
            
        Returns:
            True if the container was stopped successfully, False otherwise
        """
        self.log(f"Stopping container {container_id}...")
        exit_code, stdout, stderr = self.run_command([
            "docker", "stop", container_id
        ])
        
        if exit_code != 0:
            self.log(f"Failed to stop container: {stderr}")
            return False
            
        self.log(f"Container {container_id} stopped.")
        return True
    
    def remove_container(self, container_id: str) -> bool:
        """
        Remove a container.
        
        Args:
            container_id: ID of the container to remove
            
        Returns:
            True if the container was removed successfully, False otherwise
        """
        self.log(f"Removing container {container_id}...")
        exit_code, stdout, stderr = self.run_command([
            "docker", "rm", container_id
        ])
        
        if exit_code != 0:
            self.log(f"Failed to remove container: {stderr}")
            return False
            
        self.log(f"Container {container_id} removed.")
        return True
    
    def check_container_logs(self, container_id: str) -> str:
        """
        Check the logs of a running container.
        
        Args:
            container_id: ID of the container
            
        Returns:
            Container logs as a string
        """
        self.log(f"Checking logs for container {container_id}...")
        exit_code, stdout, stderr = self.run_command([
            "docker", "logs", container_id
        ])
        
        if exit_code != 0:
            self.log(f"Failed to get container logs: {stderr}")
            return ""
            
        if self.verbose:
            self.log(f"Container logs:\n{stdout}")
        else:
            log_lines = stdout.split("\n")
            if len(log_lines) > 10:
                self.log(f"Container logs (first 10 lines):\n" + "\n".join(log_lines[:10]) + "\n[...]")
            else:
                self.log(f"Container logs:\n{stdout}")
                
        return stdout
    
    def check_container_health(self, container_id: str) -> bool:
        """
        Check the health of a running container.
        
        Args:
            container_id: ID of the container
            
        Returns:
            True if the container is healthy, False otherwise
        """
        self.log(f"Checking health for container {container_id}...")
        exit_code, stdout, stderr = self.run_command([
            "docker", "inspect", "--format={{.State.Health.Status}}", container_id
        ])
        
        if exit_code != 0:
            self.log(f"Failed to get container health: {stderr}")
            return False
            
        health_status = stdout.strip()
        self.log(f"Container health status: {health_status}")
        
        # If container doesn't have a health check
        if health_status == "<no value>":
            # Check if container is running
            exit_code, stdout, stderr = self.run_command([
                "docker", "inspect", "--format={{.State.Status}}", container_id
            ])
            
            if exit_code != 0:
                self.log(f"Failed to get container status: {stderr}")
                return False
                
            status = stdout.strip()
            self.log(f"Container status: {status}")
            return status == "running"
            
        return health_status == "healthy"
    
    def test_container_functionality(self, container_id: str, retry_count: int = 5) -> bool:
        """
        Test the functionality of a running container.
        
        Args:
            container_id: ID of the container
            retry_count: Number of times to retry connection if it fails
            
        Returns:
            True if the container functionality tests pass, False otherwise
        """
        self.log(f"Testing functionality of container {container_id}...")
        
        # Wait for container to start and services to initialize
        self.log("Waiting for services to initialize...")
        for i in range(retry_count):
            time.sleep(5)
            
            # Try to connect to the MCP server in the container
            try:
                # Use a temporary client to connect to the container
                container_client = MCPClient(base_url="http://localhost:8080")
                status = container_client.get_status()
                
                self.log(f"Successfully connected to MCP server in container: {status}")
                
                # Test health endpoint
                health = container_client.get_health()
                self.log(f"Container health check: {health}")
                
                # If we got this far, tests passed
                return True
                
            except Exception as e:
                self.log(f"Attempt {i+1}/{retry_count}: Failed to connect to MCP server in container: {e}")
                
                if i == retry_count - 1:
                    self.log("Max retry count reached. Container functionality test failed.")
                    return False
                    
                self.log("Retrying in 5 seconds...")
        
        return False
    
    def run_full_test(self) -> bool:
        """
        Run a full test of the Docker image.
        
        This includes:
        - Verifying Docker is available
        - Pulling the image
        - Verifying the image
        - Running a container
        - Testing container functionality
        - Cleaning up
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.log(f"Starting full test of Docker image {self.full_image}")
        
        # Step 1: Verify Docker is available
        if not self.verify_docker_available():
            return False
        
        # Step 2: Check if image exists locally, pull if not
        if not self.check_image_exists_locally():
            if not self.pull_image():
                return False
        
        # Step 3: Verify the image
        if not self.verify_image():
            return False
        
        # Step 4: Run a container
        container_name = f"mcp-test-{int(time.time())}"
        container_id = self.run_container(container_name=container_name)
        if not container_id:
            return False
        
        try:
            # Step 5: Check container logs
            self.check_container_logs(container_id)
            
            # Step 6: Wait a bit and check health
            self.log("Waiting for container to initialize...")
            time.sleep(10)
            if not self.check_container_health(container_id):
                self.log("Container health check failed, but continuing with tests...")
            
            # Step 7: Test container functionality
            if not self.test_container_functionality(container_id):
                return False
                
            self.log("All container functionality tests passed!")
            return True
            
        finally:
            # Clean up
            self.log("Cleaning up...")
            self.stop_container(container_id)
            self.remove_container(container_id)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Cognee MCP Docker Image")
    parser.add_argument("--image-name", default="cognee/cognee-mcp",
                        help="Name of the Docker image to test")
    parser.add_argument("--image-tag", default="latest",
                        help="Tag of the Docker image to test")
    parser.add_argument("--mcp-url", default=os.environ.get("COGNEE_MCP_URL", "http://localhost:8080"),
                        help="URL of the MCP server")
    parser.add_argument("--api-key", default=os.environ.get("COGNEE_MCP_API_KEY"),
                        help="API key for the MCP server")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create MCP client
    client = MCPClient(
        base_url=args.mcp_url,
        api_key=args.api_key,
        debug=args.verbose
    )
    
    # Create Docker image tester
    tester = DockerImageTester(
        mcp_client=client,
        image_name=args.image_name,
        image_tag=args.image_tag,
        verbose=args.verbose
    )
    
    # Run the tests
    if tester.run_full_test():
        print("\n✅ Docker image tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Docker image tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 