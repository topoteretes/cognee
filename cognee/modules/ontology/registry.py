"""Ontology registry implementation."""

from typing import Dict, List, Optional
from uuid import uuid4

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    IOntologyRegistry,
    OntologyGraph,
    OntologyScope,
    OntologyContext,
)

logger = get_logger("OntologyRegistry")


class OntologyRegistry(IOntologyRegistry):
    """In-memory implementation of ontology registry."""

    def __init__(self):
        self.ontologies: Dict[str, OntologyGraph] = {}
        self.scope_index: Dict[OntologyScope, List[str]] = {
            scope: [] for scope in OntologyScope
        }
        self.domain_index: Dict[str, List[str]] = {}
        self.user_index: Dict[str, List[str]] = {}
        self.dataset_index: Dict[str, List[str]] = {}
        self.pipeline_index: Dict[str, List[str]] = {}

    async def register_ontology(
        self,
        ontology: OntologyGraph,
        scope: OntologyScope,
        context: Optional[OntologyContext] = None
    ) -> str:
        """Register an ontology."""
        
        ontology_id = ontology.id or str(uuid4())
        ontology.id = ontology_id
        ontology.scope = scope
        
        self.ontologies[ontology_id] = ontology
        
        # Update scope index
        if ontology_id not in self.scope_index[scope]:
            self.scope_index[scope].append(ontology_id)
        
        # Update domain index if applicable
        if context and context.domain:
            if context.domain not in self.domain_index:
                self.domain_index[context.domain] = []
            if ontology_id not in self.domain_index[context.domain]:
                self.domain_index[context.domain].append(ontology_id)
        
        # Update user index if applicable
        if context and context.user_id:
            if context.user_id not in self.user_index:
                self.user_index[context.user_id] = []
            if ontology_id not in self.user_index[context.user_id]:
                self.user_index[context.user_id].append(ontology_id)
        
        # Update dataset index if applicable
        if context and context.dataset_id:
            if context.dataset_id not in self.dataset_index:
                self.dataset_index[context.dataset_id] = []
            if ontology_id not in self.dataset_index[context.dataset_id]:
                self.dataset_index[context.dataset_id].append(ontology_id)
        
        # Update pipeline index if applicable
        if context and context.pipeline_name:
            if context.pipeline_name not in self.pipeline_index:
                self.pipeline_index[context.pipeline_name] = []
            if ontology_id not in self.pipeline_index[context.pipeline_name]:
                self.pipeline_index[context.pipeline_name].append(ontology_id)
        
        logger.info(f"Registered ontology {ontology_id} with scope {scope}")
        return ontology_id

    async def get_ontology(
        self,
        ontology_id: str,
        context: Optional[OntologyContext] = None
    ) -> Optional[OntologyGraph]:
        """Get ontology by ID."""
        return self.ontologies.get(ontology_id)

    async def find_ontologies(
        self,
        scope: Optional[OntologyScope] = None,
        domain: Optional[str] = None,
        context: Optional[OntologyContext] = None
    ) -> List[OntologyGraph]:
        """Find ontologies matching criteria."""
        
        candidate_ids = set()
        
        # Filter by scope
        if scope:
            candidate_ids.update(self.scope_index.get(scope, []))
        else:
            # If no scope specified, get all
            for scope_ids in self.scope_index.values():
                candidate_ids.update(scope_ids)
        
        # Filter by domain
        if domain:
            domain_ids = set(self.domain_index.get(domain, []))
            candidate_ids = candidate_ids.intersection(domain_ids)
        
        # Filter by context
        if context:
            if context.user_id:
                user_ids = set(self.user_index.get(context.user_id, []))
                if scope == OntologyScope.USER:
                    candidate_ids = candidate_ids.intersection(user_ids)
                else:
                    candidate_ids.update(user_ids)
            
            if context.dataset_id:
                dataset_ids = set(self.dataset_index.get(context.dataset_id, []))
                if scope == OntologyScope.DATASET:
                    candidate_ids = candidate_ids.intersection(dataset_ids)
                else:
                    candidate_ids.update(dataset_ids)
            
            if context.pipeline_name:
                pipeline_ids = set(self.pipeline_index.get(context.pipeline_name, []))
                if scope == OntologyScope.PIPELINE:
                    candidate_ids = candidate_ids.intersection(pipeline_ids)
                else:
                    candidate_ids.update(pipeline_ids)
        
        # Return matching ontologies
        result = []
        for ontology_id in candidate_ids:
            if ontology_id in self.ontologies:
                result.append(self.ontologies[ontology_id])
        
        logger.debug(f"Found {len(result)} ontologies matching criteria")
        return result

    async def unregister_ontology(
        self,
        ontology_id: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Unregister an ontology."""
        
        if ontology_id not in self.ontologies:
            return False
        
        ontology = self.ontologies[ontology_id]
        
        # Remove from all indices
        for scope_ids in self.scope_index.values():
            if ontology_id in scope_ids:
                scope_ids.remove(ontology_id)
        
        for domain_ids in self.domain_index.values():
            if ontology_id in domain_ids:
                domain_ids.remove(ontology_id)
        
        for user_ids in self.user_index.values():
            if ontology_id in user_ids:
                user_ids.remove(ontology_id)
        
        for dataset_ids in self.dataset_index.values():
            if ontology_id in dataset_ids:
                dataset_ids.remove(ontology_id)
        
        for pipeline_ids in self.pipeline_index.values():
            if ontology_id in pipeline_ids:
                pipeline_ids.remove(ontology_id)
        
        # Remove from main registry
        del self.ontologies[ontology_id]
        
        logger.info(f"Unregistered ontology {ontology_id}")
        return True

    def get_stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        return {
            "total_ontologies": len(self.ontologies),
            "global_ontologies": len(self.scope_index[OntologyScope.GLOBAL]),
            "domain_ontologies": len(self.scope_index[OntologyScope.DOMAIN]),
            "pipeline_ontologies": len(self.scope_index[OntologyScope.PIPELINE]),
            "user_ontologies": len(self.scope_index[OntologyScope.USER]),
            "dataset_ontologies": len(self.scope_index[OntologyScope.DATASET]),
            "unique_domains": len(self.domain_index),
            "unique_users": len(self.user_index),
            "unique_datasets": len(self.dataset_index),
            "unique_pipelines": len(self.pipeline_index),
        }


class DatabaseOntologyRegistry(IOntologyRegistry):
    """Database-backed ontology registry (placeholder implementation)."""

    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        # This would use actual database operations in a real implementation
        self._memory_registry = OntologyRegistry()

    async def register_ontology(
        self,
        ontology: OntologyGraph,
        scope: OntologyScope,
        context: Optional[OntologyContext] = None
    ) -> str:
        """Register an ontology in database."""
        # TODO: Implement database storage
        return await self._memory_registry.register_ontology(ontology, scope, context)

    async def get_ontology(
        self,
        ontology_id: str,
        context: Optional[OntologyContext] = None
    ) -> Optional[OntologyGraph]:
        """Get ontology from database."""
        # TODO: Implement database retrieval
        return await self._memory_registry.get_ontology(ontology_id, context)

    async def find_ontologies(
        self,
        scope: Optional[OntologyScope] = None,
        domain: Optional[str] = None,
        context: Optional[OntologyContext] = None
    ) -> List[OntologyGraph]:
        """Find ontologies in database."""
        # TODO: Implement database query
        return await self._memory_registry.find_ontologies(scope, domain, context)

    async def unregister_ontology(
        self,
        ontology_id: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Unregister ontology from database."""
        # TODO: Implement database deletion
        return await self._memory_registry.unregister_ontology(ontology_id, context)
