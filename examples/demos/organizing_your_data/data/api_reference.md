# Acme API Reference — Rate Limits

Every API key is rate limited to **100 requests per minute**. Requests beyond
the limit receive an HTTP 429 response with a `Retry-After` header.

Enterprise plans can raise the limit to **500 requests per minute per key**.
No plan, including Enterprise, offers unlimited requests.

Rate limits are enforced per key, not per account. Provision multiple keys to
shard traffic across workloads.
