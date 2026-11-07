from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx


class PolarConnector:
    """Ingests Polar.sh customers, subscriptions, and benefit grants for Cognee graph indexing."""

    DEFAULT_BASE_URL = "https://api.polar.sh/v1"

    def __init__(
        self,
        access_token: str,
        organization_id: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.access_token = access_token
        self.organization_id = organization_id
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    async def _paginate(
        self, client: httpx.AsyncClient, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        page = 1
        query = dict(params or {})
        if self.organization_id:
            query.setdefault("organization_id", self.organization_id)
        query.setdefault("limit", 100)

        while True:
            query["page"] = page
            resp = await client.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=query)
            resp.raise_for_status()
            payload = resp.json()
            items: List[Dict[str, Any]] = payload.get("items", [])
            for item in items:
                yield item
            pagination = payload.get("pagination", {})
            max_page = pagination.get("max_page", 1)
            if page >= max_page or not items:
                break
            page += 1

    async def fetch_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch customers, customer state, subscriptions, and benefit grants from Polar API."""
        customers: List[Dict[str, Any]] = []
        customer_states: List[Dict[str, Any]] = []
        subscriptions: List[Dict[str, Any]] = []
        benefit_grants: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(headers=self._headers(), timeout=30.0) as client:
            async for customer in self._paginate(client, "/customers"):
                customers.append(customer)
                cid = customer.get("id")
                if cid:
                    state_resp = await client.get(f"{self.base_url}/customers/{cid}/state")
                    if state_resp.status_code == 200:
                        customer_states.append(state_resp.json())

            async for sub in self._paginate(client, "/subscriptions"):
                subscriptions.append(sub)

            async for grant in self._paginate(client, "/benefits/grants"):
                benefit_grants.append(grant)

        return {
            "customers": customers,
            "customer_states": customer_states,
            "subscriptions": subscriptions,
            "benefit_grants": benefit_grants,
        }
