"""Health check system for cognee API."""

from io import BytesIO
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict
from enum import Enum
from pydantic import BaseModel

from cognee.version import get_cognee_version
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    status: HealthStatus
    provider: str
    response_time_ms: int
    details: str


class HealthResponse(BaseModel):
    status: HealthStatus
    timestamp: str
    version: str
    uptime: int
    components: Dict[str, ComponentHealth]


class HealthChecker:
    def __init__(self):
        self.start_time = time.time()

    async def check_relational_db(self) -> ComponentHealth:
        """Check relational database health."""
        start_time = time.time()
        try:
            from cognee.infrastructure.databases.relational.get_relational_engine import (
                get_relational_engine,
            )
            from cognee.infrastructure.databases.relational.config import get_relational_config

            config = get_relational_config()
            engine = get_relational_engine()

            # Test connection by creating a session
            session = engine.get_session()
            if session:
                session.close()

            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                provider=config.db_provider,
                response_time_ms=response_time,
                details="Connection successful",
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            logger.error(f"Relational DB health check failed: {str(e)}", exc_info=True)
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                provider="unknown",
                response_time_ms=response_time,
                details=f"Connection failed: {str(e)}",
            )

    async def check_vector_db(self) -> ComponentHealth:
        """Check vector database health."""
        start_time = time.time()
        try:
            from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
            from cognee.infrastructure.databases.vector.config import get_vectordb_config

            config = get_vectordb_config()
            engine = get_vector_engine()

            # Test basic operation - just check if engine is accessible
            if hasattr(engine, "health_check"):
                await engine.health_check()
            elif hasattr(engine, "list_tables"):
                # For LanceDB and similar
                engine.list_tables()

            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                provider=config.vector_db_provider,
                response_time_ms=response_time,
                details="Index accessible",
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            logger.error(f"Vector DB health check failed: {str(e)}", exc_info=True)
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                provider="unknown",
                response_time_ms=response_time,
                details=f"Connection failed: {str(e)}",
            )

    async def check_graph_db(self) -> ComponentHealth:
        """Check graph database health."""
        start_time = time.time()
        try:
            from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
            from cognee.infrastructure.databases.graph.config import get_graph_config

            config = get_graph_config()
            engine = await get_graph_engine()

            # Test basic operation with actual graph query
            if hasattr(engine, "query"):
                # For other graph engines
                await engine.query("MATCH () RETURN count(*) LIMIT 1", {})
            # If engine exists but no test method, consider it healthy

            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                provider=config.graph_database_provider,
                response_time_ms=response_time,
                details="Schema validated",
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            logger.error(f"Graph DB health check failed: {str(e)}", exc_info=True)
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                provider="unknown",
                response_time_ms=response_time,
                details=f"Connection failed: {str(e)}",
            )

    async def check_file_storage(self) -> ComponentHealth:
        """Check file storage health."""
        start_time = time.time()
        try:
            import os
            from cognee.infrastructure.files.storage.get_file_storage import get_file_storage
            from cognee.base_config import get_base_config

            base_config = get_base_config()
            storage = get_file_storage(base_config.data_root_directory)

            # Determine provider
            provider = "s3" if base_config.data_root_directory.startswith("s3://") else "local"

            # Test storage accessibility - for local storage, just check directory exists
            if provider == "local":
                os.makedirs(base_config.data_root_directory, exist_ok=True)
                # Simple write/read test
                test_file = os.path.join(base_config.data_root_directory, "health_check_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            else:
                # For S3, test basic operations
                test_path = "health_check_test"
                await storage.store(test_path, BytesIO(b"test"))
                await storage.remove(test_path)

            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                provider=provider,
                response_time_ms=response_time,
                details="Storage accessible",
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                provider="unknown",
                response_time_ms=response_time,
                details=f"Storage test failed: {str(e)}",
            )

    async def check_llm_provider(self) -> ComponentHealth:
        """Check LLM provider health (non-critical)."""
        start_time = time.time()
        try:
            from cognee.infrastructure.llm.config import get_llm_config

            config = get_llm_config()

            from cognee.infrastructure.llm.utils import test_llm_connection

            await test_llm_connection()

            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                provider=config.llm_provider,
                response_time_ms=response_time,
                details="API responding",
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            logger.error(f"LLM provider health check failed: {str(e)}", exc_info=True)
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                provider="unknown",
                response_time_ms=response_time,
                details=f"API check failed: {str(e)}",
            )

    async def check_embedding_service(self) -> ComponentHealth:
        """Check embedding service health (non-critical)."""
        start_time = time.time()
        try:
            from cognee.infrastructure.llm.utils import test_embedding_connection

            await test_embedding_connection()

            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                provider="configured",
                response_time_ms=response_time,
                details="Embedding generation working",
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                provider="unknown",
                response_time_ms=response_time,
                details=f"Embedding test failed: {str(e)}",
            )

    async def get_health_status(self, detailed: bool = False) -> HealthResponse:
        """Get comprehensive health status."""
        components = {}

        critical_checks = [
            ("relational_db", self.check_relational_db()),
            ("vector_db", self.check_vector_db()),
            ("graph_db", self.check_graph_db()),
            ("file_storage", self.check_file_storage()),
            ("llm_provider", self.check_llm_provider()),
            ("embedding_service", self.check_embedding_service()),
        ]

        # Non-critical services (only for detailed checks)
        non_critical_checks = []

        # Run critical checks
        critical_results = await asyncio.gather(
            *[check for _, check in critical_checks], return_exceptions=True
        )

        for (name, _), result in zip(critical_checks, critical_results):
            if isinstance(result, Exception):
                components[name] = ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    provider="unknown",
                    response_time_ms=0,
                    details=f"Health check failed: {str(result)}",
                )
            else:
                components[name] = result

        # Run non-critical checks if detailed (currently none)
        if detailed and non_critical_checks:
            non_critical_results = await asyncio.gather(
                *[check for _, check in non_critical_checks], return_exceptions=True
            )

            for (name, _), result in zip(non_critical_checks, non_critical_results):
                if isinstance(result, Exception):
                    components[name] = ComponentHealth(
                        status=HealthStatus.DEGRADED,
                        provider="unknown",
                        response_time_ms=0,
                        details=f"Health check failed: {str(result)}",
                    )
                else:
                    components[name] = result

        critical_comps = [check[0] for check in critical_checks]
        # Determine overall status
        critical_unhealthy = any(
            comp.status == HealthStatus.UNHEALTHY and name in critical_comps
            for name, comp in components.items()
        )

        has_degraded = any(comp.status == HealthStatus.DEGRADED for comp in components.values())

        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif has_degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=get_cognee_version(),
            uptime=int(time.time() - self.start_time),
            components=components,
        )


# Global health checker instance
health_checker = HealthChecker()
