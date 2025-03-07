"""
Example usage of the DataPoint-based layered knowledge graph implementation.

This script demonstrates how to create, modify, and analyze a layered knowledge graph 
using the DataPoint-based implementation for integration with Cognee infrastructure.
"""

import asyncio
import json
import logging
import os
from uuid import uuid4, UUID
from datetime import datetime

from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP
)
from cognee.shared.data_models import Node, Edge, KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CompanyKnowledgeGraphBuilder:
    """
    Builder for creating a layered knowledge graph about a company's structure.
    This demonstrates a practical application of the layered graph system.
    """
    
    def __init__(self, company_name: str):
        """Initialize the builder with a company name."""
        self.company_name = company_name
        self.graph = LayeredKnowledgeGraphDP.create_empty(
            name=f"{company_name} Organization Graph",
            description=f"Layered knowledge graph of {company_name}'s organizational structure"
        )
        
        # Track layers
        self.layer_ids = {}
    
    async def build_graph(self) -> LayeredKnowledgeGraphDP:
        """
        Build the complete layered knowledge graph.
        
        Returns:
            The built layered knowledge graph
        """
        # Create the organizational layers
        await self._create_structural_layers()
        
        # Add organizational structure nodes and edges
        await self._add_organization_structure()
        
        # Add people layer
        await self._add_people_layer()
        
        # Add project layer
        await self._add_project_layer()
        
        # Add insights layer
        await self._add_insights_layer()
        
        return self.graph
    
    async def _create_structural_layers(self):
        """Create the structural layers of the organization graph."""
        # Base organizational structure layer
        org_layer = GraphLayer(
            id=uuid4(),
            name="Organizational Structure",
            description="Basic organizational structure layer",
            layer_type="structure"
        )
        self.graph.add_layer(org_layer)
        self.layer_ids["org"] = org_layer.id
        
        # People layer (depends on org layer)
        people_layer = GraphLayer(
            id=uuid4(),
            name="People",
            description="Employee and stakeholder information",
            layer_type="people",
            parent_layers=[org_layer.id]
        )
        self.graph.add_layer(people_layer)
        self.layer_ids["people"] = people_layer.id
        
        # Projects layer (depends on org and people layers)
        projects_layer = GraphLayer(
            id=uuid4(),
            name="Projects",
            description="Project information and status",
            layer_type="projects",
            parent_layers=[org_layer.id, people_layer.id]
        )
        self.graph.add_layer(projects_layer)
        self.layer_ids["projects"] = projects_layer.id
        
        # Insights layer (depends on all previous layers)
        insights_layer = GraphLayer(
            id=uuid4(),
            name="Insights",
            description="Analytical insights and metrics",
            layer_type="insights",
            parent_layers=[org_layer.id, people_layer.id, projects_layer.id]
        )
        self.graph.add_layer(insights_layer)
        self.layer_ids["insights"] = insights_layer.id
        
        logger.info(f"Created {len(self.graph.layers)} structural layers")
    
    async def _add_organization_structure(self):
        """Add organization structure nodes and edges to the org layer."""
        org_layer_id = self.layer_ids["org"]
        
        # Create the company node
        company_node = GraphNode(
            id=uuid4(),
            name=self.company_name,
            node_type="Company",
            description=f"Main company entity: {self.company_name}"
        )
        self.graph.add_node_to_layer(company_node, org_layer_id)
        
        # Create department nodes
        departments = [
            {"name": "Engineering", "description": "Software engineering department"},
            {"name": "Product", "description": "Product management department"},
            {"name": "Design", "description": "Product design department"},
            {"name": "Marketing", "description": "Marketing and sales department"},
            {"name": "HR", "description": "Human resources department"},
            {"name": "Finance", "description": "Finance and accounting department"}
        ]
        
        department_nodes = {}
        for dept in departments:
            dept_node = GraphNode(
                id=uuid4(),
                name=dept["name"],
                node_type="Department",
                description=dept["description"]
            )
            self.graph.add_node_to_layer(dept_node, org_layer_id)
            department_nodes[dept["name"]] = dept_node
            
            # Connect department to company
            dept_edge = GraphEdge(
                source_node_id=dept_node.id,
                target_node_id=company_node.id,
                relationship_name="PART_OF"
            )
            self.graph.add_edge_to_layer(dept_edge, org_layer_id)
        
        # Create team nodes under departments
        teams = [
            {"name": "Backend", "department": "Engineering", "description": "Backend development team"},
            {"name": "Frontend", "department": "Engineering", "description": "Frontend development team"},
            {"name": "DevOps", "department": "Engineering", "description": "Infrastructure and operations team"},
            {"name": "Data Science", "department": "Engineering", "description": "Data science and analytics team"},
            {"name": "Mobile", "department": "Engineering", "description": "Mobile development team"},
            {"name": "PM", "department": "Product", "description": "Product managers team"},
            {"name": "Research", "department": "Product", "description": "User research team"},
            {"name": "UI", "department": "Design", "description": "User interface design team"},
            {"name": "UX", "department": "Design", "description": "User experience design team"},
            {"name": "Digital", "department": "Marketing", "description": "Digital marketing team"},
            {"name": "Sales", "department": "Marketing", "description": "Sales team"},
            {"name": "Recruiting", "department": "HR", "description": "Recruiting team"},
            {"name": "Operations", "department": "HR", "description": "HR operations team"},
            {"name": "Accounting", "department": "Finance", "description": "Accounting team"},
            {"name": "FP&A", "department": "Finance", "description": "Financial planning and analysis team"}
        ]
        
        team_nodes = {}
        for team in teams:
            team_node = GraphNode(
                id=uuid4(),
                name=team["name"],
                node_type="Team",
                description=team["description"]
            )
            self.graph.add_node_to_layer(team_node, org_layer_id)
            team_nodes[team["name"]] = team_node
            
            # Connect team to department
            dept_node = department_nodes[team["department"]]
            team_edge = GraphEdge(
                source_node_id=team_node.id,
                target_node_id=dept_node.id,
                relationship_name="PART_OF"
            )
            self.graph.add_edge_to_layer(team_edge, org_layer_id)
        
        # Store references for other layers
        self.company_node = company_node
        self.department_nodes = department_nodes
        self.team_nodes = team_nodes
        
        org_nodes = self.graph.get_layer_nodes(org_layer_id)
        org_edges = self.graph.get_layer_edges(org_layer_id)
        logger.info(f"Added {len(org_nodes)} nodes and {len(org_edges)} edges to organizational structure layer")
    
    async def _add_people_layer(self):
        """Add people nodes and their relationships to the people layer."""
        people_layer_id = self.layer_ids["people"]
        
        # Create executive nodes
        executives = [
            {"name": "John Smith", "title": "CEO", "department": None, "description": "Chief Executive Officer"},
            {"name": "Jane Doe", "title": "CTO", "department": "Engineering", "description": "Chief Technology Officer"},
            {"name": "Mike Johnson", "title": "CPO", "department": "Product", "description": "Chief Product Officer"},
            {"name": "Lisa Wong", "title": "CMO", "department": "Marketing", "description": "Chief Marketing Officer"},
            {"name": "David Chen", "title": "CHRO", "department": "HR", "description": "Chief HR Officer"},
            {"name": "Sarah Miller", "title": "CFO", "department": "Finance", "description": "Chief Financial Officer"}
        ]
        
        executive_nodes = {}
        for exec_info in executives:
            exec_node = GraphNode(
                id=uuid4(),
                name=exec_info["name"],
                node_type="Executive",
                description=exec_info["description"],
                properties={"title": exec_info["title"]}
            )
            self.graph.add_node_to_layer(exec_node, people_layer_id)
            executive_nodes[exec_info["name"]] = exec_node
            
            # Connect exec to company
            if exec_info["department"] is None:
                # CEO connects directly to company
                exec_edge = GraphEdge(
                    source_node_id=exec_node.id,
                    target_node_id=self.company_node.id,
                    relationship_name="LEADS"
                )
                self.graph.add_edge_to_layer(exec_edge, people_layer_id)
            else:
                # Other execs connect to their departments
                dept_node = self.department_nodes[exec_info["department"]]
                exec_edge = GraphEdge(
                    source_node_id=exec_node.id,
                    target_node_id=dept_node.id,
                    relationship_name="LEADS"
                )
                self.graph.add_edge_to_layer(exec_edge, people_layer_id)
        
        # Create manager nodes
        managers = [
            {"name": "Robert Taylor", "title": "Engineering Manager", "team": "Backend", "description": "Backend Engineering Manager"},
            {"name": "Emma Lewis", "title": "Engineering Manager", "team": "Frontend", "description": "Frontend Engineering Manager"},
            {"name": "Alex Rivera", "title": "Engineering Manager", "team": "DevOps", "description": "DevOps Engineering Manager"},
            {"name": "Jennifer Park", "title": "Data Science Manager", "team": "Data Science", "description": "Data Science Manager"},
            {"name": "Tom Garcia", "title": "Engineering Manager", "team": "Mobile", "description": "Mobile Engineering Manager"},
            {"name": "Olivia Brown", "title": "Product Manager", "team": "PM", "description": "Senior Product Manager"},
            {"name": "Nathan Wilson", "title": "Research Manager", "team": "Research", "description": "User Research Manager"},
            {"name": "Sophia Lee", "title": "Design Manager", "team": "UI", "description": "UI Design Manager"},
            {"name": "Ethan Davis", "title": "Design Manager", "team": "UX", "description": "UX Design Manager"},
            {"name": "Ava Martinez", "title": "Marketing Manager", "team": "Digital", "description": "Digital Marketing Manager"},
            {"name": "Noah Clark", "title": "Sales Manager", "team": "Sales", "description": "Sales Manager"},
            {"name": "Isabella Walker", "title": "Recruiting Manager", "team": "Recruiting", "description": "Recruiting Manager"},
            {"name": "Mason Hall", "title": "HR Operations Manager", "team": "Operations", "description": "HR Operations Manager"},
            {"name": "Charlotte White", "title": "Accounting Manager", "team": "Accounting", "description": "Accounting Manager"},
            {"name": "Liam Scott", "title": "Financial Analyst Manager", "team": "FP&A", "description": "Financial Planning Manager"}
        ]
        
        manager_nodes = {}
        for mgr_info in managers:
            mgr_node = GraphNode(
                id=uuid4(),
                name=mgr_info["name"],
                node_type="Manager",
                description=mgr_info["description"],
                properties={"title": mgr_info["title"]}
            )
            self.graph.add_node_to_layer(mgr_node, people_layer_id)
            manager_nodes[mgr_info["name"]] = mgr_node
            
            # Connect manager to team
            team_node = self.team_nodes[mgr_info["team"]]
            mgr_edge = GraphEdge(
                source_node_id=mgr_node.id,
                target_node_id=team_node.id,
                relationship_name="LEADS"
            )
            self.graph.add_edge_to_layer(mgr_edge, people_layer_id)
        
        # Connect executives to managers (reporting relationships)
        reporting_relationships = {
            "Jane Doe": ["Robert Taylor", "Emma Lewis", "Alex Rivera", "Jennifer Park", "Tom Garcia"],
            "Mike Johnson": ["Olivia Brown", "Nathan Wilson"],
            "Lisa Wong": ["Ava Martinez", "Noah Clark"],
            "David Chen": ["Isabella Walker", "Mason Hall"],
            "Sarah Miller": ["Charlotte White", "Liam Scott"]
        }
        
        for exec_name, manager_list in reporting_relationships.items():
            exec_node = executive_nodes[exec_name]
            for mgr_name in manager_list:
                mgr_node = manager_nodes[mgr_name]
                report_edge = GraphEdge(
                    source_node_id=mgr_node.id,
                    target_node_id=exec_node.id,
                    relationship_name="REPORTS_TO"
                )
                self.graph.add_edge_to_layer(report_edge, people_layer_id)
        
        # Store references for other layers
        self.executive_nodes = executive_nodes
        self.manager_nodes = manager_nodes
        
        people_nodes = self.graph.get_layer_nodes(people_layer_id)
        people_edges = self.graph.get_layer_edges(people_layer_id)
        logger.info(f"Added {len(people_nodes)} nodes and {len(people_edges)} edges to people layer")
    
    async def _add_project_layer(self):
        """Add project nodes and their relationships to the project layer."""
        projects_layer_id = self.layer_ids["projects"]
        
        # Create project nodes
        projects = [
            {
                "name": "Mobile App Redesign",
                "description": "Redesign of the mobile app UI/UX",
                "status": "In Progress",
                "start_date": "2023-01-15",
                "teams": ["Mobile", "UI", "UX"],
                "manager": "Tom Garcia"
            },
            {
                "name": "Backend API Refactoring",
                "description": "Refactoring of the core API services",
                "status": "In Progress",
                "start_date": "2023-02-01",
                "teams": ["Backend", "DevOps"],
                "manager": "Robert Taylor"
            },
            {
                "name": "Data Analytics Platform",
                "description": "Building a new data analytics platform",
                "status": "Planning",
                "start_date": "2023-03-15",
                "teams": ["Data Science", "Backend", "Frontend"],
                "manager": "Jennifer Park"
            },
            {
                "name": "Customer Portal",
                "description": "New customer self-service portal",
                "status": "In Progress",
                "start_date": "2023-01-10",
                "teams": ["Frontend", "Backend", "UI", "UX"],
                "manager": "Emma Lewis"
            },
            {
                "name": "Cloud Migration",
                "description": "Migration of infrastructure to cloud",
                "status": "In Progress",
                "start_date": "2022-11-01",
                "teams": ["DevOps", "Backend"],
                "manager": "Alex Rivera"
            },
            {
                "name": "Marketing Campaign",
                "description": "Q1 digital marketing campaign",
                "status": "Completed",
                "start_date": "2023-01-01",
                "end_date": "2023-03-31",
                "teams": ["Digital", "Sales"],
                "manager": "Ava Martinez"
            }
        ]
        
        project_nodes = {}
        for project in projects:
            project_node = GraphNode(
                id=uuid4(),
                name=project["name"],
                node_type="Project",
                description=project["description"],
                properties={
                    "status": project["status"],
                    "start_date": project["start_date"],
                    "end_date": project.get("end_date")
                }
            )
            self.graph.add_node_to_layer(project_node, projects_layer_id)
            project_nodes[project["name"]] = project_node
            
            # Connect project to teams
            for team_name in project["teams"]:
                team_node = self.team_nodes[team_name]
                team_edge = GraphEdge(
                    source_node_id=team_node.id,
                    target_node_id=project_node.id,
                    relationship_name="WORKS_ON"
                )
                self.graph.add_edge_to_layer(team_edge, projects_layer_id)
            
            # Connect project to manager
            mgr_node = self.manager_nodes[project["manager"]]
            mgr_edge = GraphEdge(
                source_node_id=mgr_node.id,
                target_node_id=project_node.id,
                relationship_name="MANAGES"
            )
            self.graph.add_edge_to_layer(mgr_edge, projects_layer_id)
        
        # Create dependencies between projects
        dependencies = [
            {"from": "Backend API Refactoring", "to": "Customer Portal", "description": "API needed for portal"},
            {"from": "Backend API Refactoring", "to": "Mobile App Redesign", "description": "API needed for mobile"},
            {"from": "Cloud Migration", "to": "Data Analytics Platform", "description": "Platform will use new cloud infrastructure"}
        ]
        
        for dep in dependencies:
            source_node = project_nodes[dep["from"]]
            target_node = project_nodes[dep["to"]]
            dep_edge = GraphEdge(
                source_node_id=source_node.id,
                target_node_id=target_node.id,
                relationship_name="DEPENDENCY",
                properties={"description": dep["description"]}
            )
            self.graph.add_edge_to_layer(dep_edge, projects_layer_id)
        
        # Store references for other layers
        self.project_nodes = project_nodes
        
        project_nodes = self.graph.get_layer_nodes(projects_layer_id)
        project_edges = self.graph.get_layer_edges(projects_layer_id)
        logger.info(f"Added {len(project_nodes)} nodes and {len(project_edges)} edges to projects layer")
    
    async def _add_insights_layer(self):
        """Add insights nodes and their relationships to the insights layer."""
        insights_layer_id = self.layer_ids["insights"]
        
        # Create risk nodes
        risks = [
            {
                "name": "Schedule Risk: Cloud Migration",
                "description": "Risk of schedule slippage in cloud migration",
                "severity": "High",
                "impact": "Might delay dependent projects",
                "project": "Cloud Migration"
            },
            {
                "name": "Resource Risk: Backend API",
                "description": "Risk of resource shortage for API refactoring",
                "severity": "Medium",
                "impact": "Might require additional staffing",
                "project": "Backend API Refactoring"
            },
            {
                "name": "Technical Risk: Data Platform",
                "description": "Risk of technical challenges in data platform",
                "severity": "Medium",
                "impact": "Might require architecture changes",
                "project": "Data Analytics Platform"
            }
        ]
        
        risk_nodes = {}
        for risk in risks:
            risk_node = GraphNode(
                id=uuid4(),
                name=risk["name"],
                node_type="Risk",
                description=risk["description"],
                properties={
                    "severity": risk["severity"],
                    "impact": risk["impact"]
                }
            )
            self.graph.add_node_to_layer(risk_node, insights_layer_id)
            risk_nodes[risk["name"]] = risk_node
            
            # Connect risk to project
            project_node = self.project_nodes[risk["project"]]
            risk_edge = GraphEdge(
                source_node_id=risk_node.id,
                target_node_id=project_node.id,
                relationship_name="AFFECTS"
            )
            self.graph.add_edge_to_layer(risk_edge, insights_layer_id)
        
        # Create metrics nodes
        metrics = [
            {
                "name": "Team Velocity: Backend",
                "description": "Sprint velocity for Backend team",
                "value": 42,
                "unit": "Story Points",
                "team": "Backend"
            },
            {
                "name": "Team Velocity: Frontend",
                "description": "Sprint velocity for Frontend team",
                "value": 38,
                "unit": "Story Points",
                "team": "Frontend"
            },
            {
                "name": "Team Velocity: Mobile",
                "description": "Sprint velocity for Mobile team",
                "value": 35,
                "unit": "Story Points",
                "team": "Mobile"
            },
            {
                "name": "Project Burndown: Customer Portal",
                "description": "Burndown chart for Customer Portal",
                "value": 65,
                "unit": "Percent Complete",
                "project": "Customer Portal"
            },
            {
                "name": "Project Burndown: Mobile App Redesign",
                "description": "Burndown chart for Mobile App Redesign",
                "value": 40,
                "unit": "Percent Complete",
                "project": "Mobile App Redesign"
            }
        ]
        
        metric_nodes = {}
        for metric in metrics:
            metric_node = GraphNode(
                id=uuid4(),
                name=metric["name"],
                node_type="Metric",
                description=metric["description"],
                properties={
                    "value": metric["value"],
                    "unit": metric["unit"],
                    "updated_at": datetime.now().isoformat()
                }
            )
            self.graph.add_node_to_layer(metric_node, insights_layer_id)
            metric_nodes[metric["name"]] = metric_node
            
            # Connect metric to team or project
            if "team" in metric:
                team_node = self.team_nodes[metric["team"]]
                metric_edge = GraphEdge(
                    source_node_id=metric_node.id,
                    target_node_id=team_node.id,
                    relationship_name="MEASURES"
                )
                self.graph.add_edge_to_layer(metric_edge, insights_layer_id)
            elif "project" in metric:
                project_node = self.project_nodes[metric["project"]]
                metric_edge = GraphEdge(
                    source_node_id=metric_node.id,
                    target_node_id=project_node.id,
                    relationship_name="MEASURES"
                )
                self.graph.add_edge_to_layer(metric_edge, insights_layer_id)
        
        # Create recommendations
        recommendations = [
            {
                "name": "Resource Allocation Recommendation",
                "description": "Recommend adding resources to Backend team",
                "action": "Add 2 senior engineers to Backend team",
                "target_team": "Backend",
                "related_risk": "Resource Risk: Backend API"
            },
            {
                "name": "Schedule Adjustment Recommendation",
                "description": "Recommend adjusting Cloud Migration timeline",
                "action": "Extend Cloud Migration deadline by 2 weeks",
                "target_project": "Cloud Migration",
                "related_risk": "Schedule Risk: Cloud Migration"
            }
        ]
        
        recommendation_nodes = {}
        for rec in recommendations:
            rec_node = GraphNode(
                id=uuid4(),
                name=rec["name"],
                node_type="Recommendation",
                description=rec["description"],
                properties={
                    "action": rec["action"],
                    "created_at": datetime.now().isoformat()
                }
            )
            self.graph.add_node_to_layer(rec_node, insights_layer_id)
            recommendation_nodes[rec["name"]] = rec_node
            
            # Connect recommendation to team or project
            if "target_team" in rec:
                team_node = self.team_nodes[rec["target_team"]]
                rec_edge = GraphEdge(
                    source_node_id=rec_node.id,
                    target_node_id=team_node.id,
                    relationship_name="TARGETS"
                )
                self.graph.add_edge_to_layer(rec_edge, insights_layer_id)
            elif "target_project" in rec:
                project_node = self.project_nodes[rec["target_project"]]
                rec_edge = GraphEdge(
                    source_node_id=rec_node.id,
                    target_node_id=project_node.id,
                    relationship_name="TARGETS"
                )
                self.graph.add_edge_to_layer(rec_edge, insights_layer_id)
            
            # Connect recommendation to risk
            risk_node = risk_nodes[rec["related_risk"]]
            risk_edge = GraphEdge(
                source_node_id=rec_node.id,
                target_node_id=risk_node.id,
                relationship_name="ADDRESSES"
            )
            self.graph.add_edge_to_layer(risk_edge, insights_layer_id)
        
        insights_nodes = self.graph.get_layer_nodes(insights_layer_id)
        insights_edges = self.graph.get_layer_edges(insights_layer_id)
        logger.info(f"Added {len(insights_nodes)} nodes and {len(insights_edges)} edges to insights layer")


async def analyze_layered_graph(graph: LayeredKnowledgeGraphDP):
    """
    Analyze the layered knowledge graph and print insights.
    
    Args:
        graph: The layered knowledge graph to analyze
    """
    # Get metrics for each layer
    logger.info("\n=== Layer Metrics ===")
    for layer_id in graph.layers:
        layer = graph._get_layer(layer_id)
        layer_nodes = graph.get_layer_nodes(layer_id)
        layer_edges = graph.get_layer_edges(layer_id)
        
        # Get node type distribution
        node_types = {}
        for node in layer_nodes:
            if node.node_type not in node_types:
                node_types[node.node_type] = 0
            node_types[node.node_type] += 1
        
        # Get edge type distribution
        edge_types = {}
        for edge in layer_edges:
            if edge.relationship_name not in edge_types:
                edge_types[edge.relationship_name] = 0
            edge_types[edge.relationship_name] += 1
        
        logger.info(f"\nLayer: {layer.name} (ID: {layer.id})")
        logger.info(f"  Nodes: {len(layer_nodes)}")
        logger.info(f"  Edges: {len(layer_edges)}")
        logger.info("  Node types:")
        for node_type, count in node_types.items():
            logger.info(f"    - {node_type}: {count}")
        logger.info("  Edge types:")
        for edge_type, count in edge_types.items():
            logger.info(f"    - {edge_type}: {count}")
    
    # Get cumulative metrics
    logger.info("\n=== Cumulative Metrics ===")
    for i, layer_id in enumerate(graph.layers):
        layer = graph._get_layer(layer_id)
        cumulative_graph = graph.get_cumulative_layer_graph(layer_id)
        
        logger.info(f"\nCumulative up to {layer.name}:")
        logger.info(f"  Total nodes: {len(cumulative_graph.nodes)}")
        logger.info(f"  Total edges: {len(cumulative_graph.edges)}")
    
    # Analyze insights layer specifically
    if "insights" in [graph._get_layer(lid).name for lid in graph.layers]:
        insights_layer_id = next(lid for lid in graph.layers if graph._get_layer(lid).name == "Insights")
        insights_graph = graph.get_layer_graph(insights_layer_id)
        
        logger.info("\n=== Insights Analysis ===")
        
        # Extract risks
        risks = [node for node in insights_graph.nodes if node.type == "Risk"]
        logger.info(f"\nIdentified {len(risks)} risks:")
        for risk in risks:
            logger.info(f"  - {risk.name} (Severity: {risk.properties.get('severity')})")
            # Find affected projects
            for edge in insights_graph.edges:
                if edge.source_node_id == risk.id and edge.relationship_name == "AFFECTS":
                    affected_project = next((n for n in insights_graph.nodes if n.id == edge.target_node_id), None)
                    if affected_project:
                        logger.info(f"    Affects: {affected_project.name}")
        
        # Extract recommendations
        recommendations = [node for node in insights_graph.nodes if node.type == "Recommendation"]
        logger.info(f"\nIdentified {len(recommendations)} recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec.name}")
            logger.info(f"    Action: {rec.properties.get('action')}")
            
            # Find targets and addressed risks
            targets = []
            addresses = []
            for edge in insights_graph.edges:
                if edge.source_node_id == rec.id:
                    if edge.relationship_name == "TARGETS":
                        target = next((n for n in insights_graph.nodes if n.id == edge.target_node_id), None)
                        if target:
                            targets.append(target.name)
                    elif edge.relationship_name == "ADDRESSES":
                        risk = next((n for n in insights_graph.nodes if n.id == edge.target_node_id), None)
                        if risk:
                            addresses.append(risk.name)
            
            if targets:
                logger.info(f"    Targets: {', '.join(targets)}")
            if addresses:
                logger.info(f"    Addresses: {', '.join(addresses)}")


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that can handle UUID objects."""
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


async def save_graph(graph: LayeredKnowledgeGraphDP, filename: str):
    """
    Save a layered knowledge graph to a JSON file.
    
    Args:
        graph: The layered knowledge graph to save
        filename: The filename to save to
    """
    # Use the to_json method which handles serialization correctly
    json_str = graph.to_json()
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(json_str)
    
    logger.info(f"Graph saved to {filename}")


async def load_graph(filename: str) -> LayeredKnowledgeGraphDP:
    """
    Load a layered knowledge graph from a JSON file.
    
    Args:
        filename: The filename to load from
        
    Returns:
        The loaded layered knowledge graph
    """
    # Load from file
    with open(filename, 'r') as f:
        json_str = f.read()
    
    # Deserialize the graph
    graph = LayeredKnowledgeGraphDP.from_json(json_str)
    
    logger.info(f"Graph loaded from {filename}")
    return graph


async def export_visualization_data(graph: LayeredKnowledgeGraphDP, filename: str):
    """
    Export graph data in a format suitable for visualization.
    
    Args:
        graph: The graph to export
        filename: The filename to save to
    """
    # For each layer, get its graph and collect data
    layers_data = []
    nodes_data = []
    edges_data = []
    
    # Add layer data
    for layer_id in graph.layers:
        layer = graph.get_layer(layer_id)
        
        # Get parent layer names
        parent_names = []
        for parent_id in layer.parent_layers:
            parent = graph.get_layer(parent_id)
            parent_names.append(parent.name)
        
        layers_data.append({
            "id": str(layer.id),
            "name": layer.name,
            "description": layer.description,
            "type": layer.layer_type,
            "parents": parent_names
        })
    
    # Add node data (using the full graph)
    node_layer_map = {}
    for layer_id in graph.layers:
        for node in graph.get_layer_nodes(layer_id):
            node_layer_map[str(node.id)] = {"layer_id": str(layer_id), "layer_name": graph.get_layer(layer_id).name}
    
    # Get all nodes from the last layer's cumulative graph
    layer_ids = list(graph.layers.keys())
    if layer_ids:
        last_layer_id = layer_ids[-1]
        cumulative_graph = graph.get_cumulative_layer_graph(last_layer_id)
        
        for node in cumulative_graph.nodes:
            layer_info = node_layer_map.get(node.id, {"layer_id": "unknown", "layer_name": "Unknown"})
            
            nodes_data.append({
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "layer": layer_info["layer_name"],
                "description": node.description,
                "properties": node.properties if hasattr(node, "properties") else {}
            })
        
        # Add edge data
        for edge in cumulative_graph.edges:
            # Get layer info
            layer_name = "Unknown"
            if hasattr(edge, "layer_id") and edge.layer_id:
                layer_id_str = str(edge.layer_id)
                if layer_id_str in node_layer_map:
                    layer_name = node_layer_map[layer_id_str]["layer_name"]
            
            edges_data.append({
                "source": edge.source_node_id,
                "target": edge.target_node_id,
                "relationship": edge.relationship_name,
                "layer": layer_name,
                "properties": edge.properties if hasattr(edge, "properties") else {}
            })
    
    # Build the complete visualization data
    viz_data = {
        "layers": layers_data,
        "nodes": nodes_data,
        "edges": edges_data
    }
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(viz_data, f, indent=2, cls=UUIDEncoder)
    
    logger.info(f"Exported visualization data to {filename} ({len(nodes_data)} nodes, {len(edges_data)} edges)")


async def main():
    """Main function."""
    logger.info("=== DataPoint-based Layered Knowledge Graph Example ===")
    
    # Build company knowledge graph
    logger.info("\nBuilding company knowledge graph...")
    builder = CompanyKnowledgeGraphBuilder("Acme Tech")
    graph = await builder.build_graph()
    
    # Analyze the graph
    logger.info("\nAnalyzing graph...")
    await analyze_layered_graph(graph)
    
    # Save the graph
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    graph_file = os.path.join(output_dir, "company_graph.json")
    await save_graph(graph, graph_file)
    
    # Export visualization data
    viz_file = os.path.join(output_dir, "company_graph_viz.json")
    await export_visualization_data(graph, viz_file)
    
    # Test loading the graph back
    loaded_graph = await load_graph(graph_file)
    logger.info(f"\nSuccessfully loaded graph with {len(loaded_graph.layers)} layers")
    
    # Demonstrate serialization capabilities from DataPoint (if the graph has nodes)
    if loaded_graph.nodes:
        node = next(iter(loaded_graph.nodes.values()))
        logger.info(f"\nDataPoint serialization example (node {node.name}):")
        logger.info(f"  to_json(): {len(node.to_json())} characters")
        logger.info(f"  to_pickle(): {len(node.to_pickle())} bytes")
    
    logger.info("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main()) 