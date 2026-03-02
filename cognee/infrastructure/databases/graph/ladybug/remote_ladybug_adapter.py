"""Adapter for remote Kuzu graph database via REST API."""

from cognee.shared.logging_utils import get_logger
import json
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
from uuid import UUID

from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
from cognee.shared.utils import create_secure_ssl_context

logger = get_logger()


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles UUID objects."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class RemoteKuzuAdapter(KuzuAdapter):
    """Adapter for remote Kuzu graph database operations via REST API."""

    def __init__(self, api_url: str, username: str, password: str):
        """Initialize remote Kuzu database connection.

        Args:
            api_url: URL of the Kuzu REST API
            username: Optional username for API authentication
            password: Optional password for API authentication
        """
        # Initialize parent with a dummy path since we're using REST API
        super().__init__("/tmp/kuzu_remote")
        self.api_url = api_url
        self.username = username
        self.password = password
        self._session = None
        self._schema_initialized = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            ssl_context = create_secure_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        """Close the adapter and its session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _make_request(self, endpoint: str, data: dict) -> dict:
        """Make a request to the Kuzu API."""
        url = f"{self.api_url}{endpoint}"
        session = await self._get_session()
        try:
            # Use custom encoder for UUID serialization
            json_data = json.dumps(data, cls=UUIDEncoder)
            async with session.post(
                url, data=json_data, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logger.error(
                        f"API request failed with status {response.status}: {error_detail}\n"
                        f"Request data: {data}"
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_detail,
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            logger.error(f"Request data: {data}")
            raise

    async def query(self, query: str, params: Optional[dict] = None) -> List[Tuple]:
        """Execute a Kuzu query via the REST API."""
        try:
            # Initialize schema if needed
            if not self._schema_initialized:
                await self._initialize_schema()

            response = await self._make_request(
                "/query", {"query": query, "parameters": params or {}}
            )

            # Convert response to list of tuples
            results = []
            if "data" in response:
                for row in response["data"]:
                    processed_row = []
                    for val in row:
                        if isinstance(val, dict) and "properties" in val:
                            try:
                                props = json.loads(val["properties"])
                                val.update(props)
                                del val["properties"]
                            except json.JSONDecodeError:
                                pass
                        processed_row.append(val)
                    results.append(tuple(processed_row))

            return results
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {params}")
            raise

    async def _check_schema_exists(self) -> bool:
        """Check if the required schema exists without causing recursion."""
        try:
            # Make a direct request to check schema using Cypher
            response = await self._make_request(
                "/query",
                {"query": "MATCH (n:Node) RETURN COUNT(n) > 0", "parameters": {}},
            )
            return bool(response.get("data") and response["data"][0][0])
        except Exception as e:
            logger.error(f"Failed to check schema: {e}")
            return False

    async def _create_schema(self):
        """Create the required schema tables."""
        try:
            # Create Node table if it doesn't exist
            try:
                await self._make_request(
                    "/query",
                    {
                        "query": """
                        CREATE NODE TABLE IF NOT EXISTS Node (
                            id STRING,
                            name STRING,
                            type STRING,
                            properties STRING,
                            created_at TIMESTAMP,
                            updated_at TIMESTAMP,
                            PRIMARY KEY (id)
                        )
                        """,
                        "parameters": {},
                    },
                )
            except aiohttp.ClientResponseError as e:
                if "already exists" not in str(e):
                    raise

            # Create EDGE table if it doesn't exist
            try:
                await self._make_request(
                    "/query",
                    {
                        "query": """
                        CREATE REL TABLE IF NOT EXISTS EDGE (
                            FROM Node TO Node,
                            relationship_name STRING,
                            properties STRING,
                            created_at TIMESTAMP,
                            updated_at TIMESTAMP
                        )
                        """,
                        "parameters": {},
                    },
                )
            except aiohttp.ClientResponseError as e:
                if "already exists" not in str(e):
                    raise

            self._schema_initialized = True
            logger.info("Schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    async def _initialize_schema(self):
        """Initialize the database schema if it doesn't exist."""
        if self._schema_initialized:
            return

        try:
            if not await self._check_schema_exists():
                await self._create_schema()
            else:
                self._schema_initialized = True
                logger.info("Schema already exists")

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
