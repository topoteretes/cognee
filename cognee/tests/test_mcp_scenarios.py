#!/usr/bin/env python3
"""
Test scenarios for the Cognee MCP (Management Control Plane) client.

This script demonstrates various testing scenarios for the MCP server,
including service management, resource provisioning, and monitoring.
"""

import os
import json
import time
import argparse
import asyncio
from typing import Dict, Any, List
from mcp_client import MCPClient

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")

async def test_basic_connectivity(client: MCPClient):
    """Test basic connectivity to the MCP server."""
    print_section("TESTING BASIC CONNECTIVITY")
    
    try:
        # Check server status
        print("Checking server status...")
        status = client.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Check server health
        print("\nChecking server health...")
        health = client.get_health()
        print(f"Health: {json.dumps(health, indent=2)}")
        
        return True
    except Exception as e:
        print(f"Connectivity test failed: {e}")
        return False

async def test_config_management(client: MCPClient):
    """Test configuration management."""
    print_section("TESTING CONFIGURATION MANAGEMENT")
    
    try:
        # Get current configuration
        print("Getting current configuration...")
        config = client.get_config()
        print(f"Current config: {json.dumps(config, indent=2)}")
        
        # If we have a configuration, try updating a non-critical setting
        if config and isinstance(config, dict):
            # Create a backup of the original config
            original_config = config.copy()
            
            # Find a safe setting to modify (logging level is usually safe)
            if "logging" in config and "level" in config["logging"]:
                new_level = "DEBUG" if config["logging"]["level"] != "DEBUG" else "INFO"
                print(f"\nUpdating logging level to {new_level}...")
                
                update_data = {"logging": {"level": new_level}}
                updated_config = client.update_config(update_data)
                print(f"Updated config: {json.dumps(updated_config, indent=2)}")
                
                # Restore original configuration
                print("\nRestoring original configuration...")
                client.update_config({"logging": {"level": original_config["logging"]["level"]}})
            else:
                print("No safe configuration settings found to modify. Skipping update test.")
        
        return True
    except Exception as e:
        print(f"Configuration management test failed: {e}")
        return False

async def test_service_management(client: MCPClient):
    """Test service management."""
    print_section("TESTING SERVICE MANAGEMENT")
    
    try:
        # Get list of services
        print("Getting list of services...")
        services = client.get_services()
        print(f"Services: {json.dumps(services, indent=2)}")
        
        if not services:
            print("No services found. Skipping service management tests.")
            return True
        
        # Get details of the first service
        service_id = services[0].get("id", "")
        if not service_id:
            print("No valid service ID found. Skipping service management tests.")
            return True
        
        print(f"\nGetting details for service {service_id}...")
        service_details = client.get_service(service_id)
        print(f"Service details: {json.dumps(service_details, indent=2)}")
        
        # Only test start/stop for services that support it and aren't critical
        if service_details.get("can_restart", False) and not service_details.get("critical", True):
            # Restart service
            print(f"\nRestarting service {service_id}...")
            restart_result = client.restart_service(service_id)
            print(f"Restart result: {json.dumps(restart_result, indent=2)}")
            
            # Wait for service to stabilize
            print("Waiting for service to stabilize...")
            time.sleep(5)
            
            # Check service status after restart
            service_after_restart = client.get_service(service_id)
            print(f"Service status after restart: {service_after_restart.get('status', 'unknown')}")
        else:
            print(f"Service {service_id} does not support restart or is critical. Skipping restart test.")
        
        return True
    except Exception as e:
        print(f"Service management test failed: {e}")
        return False

async def test_resource_management(client: MCPClient):
    """Test resource management."""
    print_section("TESTING RESOURCE MANAGEMENT")
    
    try:
        # Get list of resources
        print("Getting list of resources...")
        resources = client.get_resources()
        print(f"Resources: {json.dumps(resources, indent=2)}")
        
        # Filter resources by type (if available)
        resource_types = set()
        for resource in resources:
            if "type" in resource:
                resource_types.add(resource["type"])
        
        for resource_type in resource_types:
            print(f"\nGetting resources of type {resource_type}...")
            typed_resources = client.get_resources(resource_type=resource_type)
            print(f"Found {len(typed_resources)} resources of type {resource_type}")
        
        # Get details of a specific resource (if available)
        if resources:
            resource_id = resources[0].get("id", "")
            if resource_id:
                print(f"\nGetting details for resource {resource_id}...")
                resource_details = client.get_resource(resource_id)
                print(f"Resource details: {json.dumps(resource_details, indent=2)}")
        
        return True
    except Exception as e:
        print(f"Resource management test failed: {e}")
        return False

async def test_monitoring(client: MCPClient):
    """Test monitoring capabilities."""
    print_section("TESTING MONITORING CAPABILITIES")
    
    try:
        # Get system logs
        print("Getting system logs...")
        logs = client.get_logs(lines=10)
        if "logs" in logs:
            print(f"Recent logs ({len(logs['logs'])} entries):")
            for log in logs["logs"][:5]:  # Show only first 5 logs
                print(f"  {log}")
        else:
            print(f"Logs: {json.dumps(logs, indent=2)}")
        
        # Get system metrics
        print("\nGetting system metrics...")
        metrics = client.get_metrics(duration="15m")
        print(f"System metrics: {json.dumps(metrics, indent=2)}")
        
        # Get service-specific metrics if services are available
        services = client.get_services()
        if services:
            service_id = services[0].get("id", "")
            if service_id:
                print(f"\nGetting metrics for service {service_id}...")
                service_metrics = client.get_metrics(service_id=service_id, duration="15m")
                print(f"Service metrics: {json.dumps(service_metrics, indent=2)}")
        
        return True
    except Exception as e:
        print(f"Monitoring test failed: {e}")
        return False

async def test_user_management(client: MCPClient):
    """Test user management capabilities."""
    print_section("TESTING USER MANAGEMENT")
    
    try:
        # Get list of users
        print("Getting list of users...")
        users = client.get_users()
        print(f"Users: {json.dumps(users, indent=2)}")
        
        # We won't create or modify users in a test to avoid potential security issues
        if users:
            user_id = users[0].get("id", "")
            if user_id:
                print(f"\nGetting details for user {user_id}...")
                user_details = client.get_user(user_id)
                print(f"User details: {json.dumps(user_details, indent=2)}")
        
        return True
    except Exception as e:
        print(f"User management test failed: {e}")
        return False

async def test_docker_image_management(client: MCPClient):
    """Test Docker image management capabilities."""
    print_section("TESTING DOCKER IMAGE MANAGEMENT")
    
    try:
        # Get list of available Docker images
        print("Getting list of available Docker images...")
        images = client.get_images()
        print(f"Images: {json.dumps(images, indent=2)}")
        
        # We won't pull images in a test to avoid unnecessary bandwidth usage
        
        return True
    except Exception as e:
        print(f"Docker image management test failed: {e}")
        return False

async def run_all_tests(client: MCPClient):
    """Run all MCP tests."""
    tests = [
        test_basic_connectivity,
        test_config_management,
        test_service_management,
        test_resource_management,
        test_monitoring,
        test_user_management,
        test_docker_image_management
    ]
    
    results = {}
    
    for test in tests:
        test_name = test.__name__
        print(f"\nRunning test: {test_name}")
        try:
            result = await test(client)
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results[test_name] = "FAILED"
    
    print_section("TEST RESULTS")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the Cognee MCP Client")
    parser.add_argument("--url", default=os.environ.get("COGNEE_MCP_URL", "http://localhost:8080"),
                        help="MCP server URL")
    parser.add_argument("--api-key", default=os.environ.get("COGNEE_MCP_API_KEY"),
                        help="API key for authentication")
    parser.add_argument("--test", choices=["all", "connectivity", "config", "services", 
                                          "resources", "monitoring", "users", "images"],
                        default="all", help="Specific test to run")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Create MCP client
    client = MCPClient(
        base_url=args.url,
        api_key=args.api_key,
        timeout=args.timeout
    )
    
    # Run specified test
    if args.test == "all":
        asyncio.run(run_all_tests(client))
    elif args.test == "connectivity":
        asyncio.run(test_basic_connectivity(client))
    elif args.test == "config":
        asyncio.run(test_config_management(client))
    elif args.test == "services":
        asyncio.run(test_service_management(client))
    elif args.test == "resources":
        asyncio.run(test_resource_management(client))
    elif args.test == "monitoring":
        asyncio.run(test_monitoring(client))
    elif args.test == "users":
        asyncio.run(test_user_management(client))
    elif args.test == "images":
        asyncio.run(test_docker_image_management(client))

if __name__ == "__main__":
    main() 