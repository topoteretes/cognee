# Sessions and Caching

> Understanding how Cognee maintains conversational memory through sessions and cache adapters

In Cognee, a session defines the scope for a single conversation or agent run. It maintains a cache of short-term information, including recent queries, responses, and the context used to answer them.

## What Is a Session?

A session is Cognee's short-term memory for a specific user during search operations. It is identified by `(user_id, session_id)` and stores an ordered list of recent interactions created by calls to `cognee.search()`.

Each interaction is stored as a dictionary containing:

* `time` – when the turn was created (UTC, ISO format)
* `question` – the user's query
* `context` – summarized context Cognee used to answer
* `answer` – the model's response

Cognee reads from this session memory at the start of a search to recover earlier turns. When the search finishes, it writes a new interaction to the session so the history grows over time.

Using the same `session_id` across searches allows Cognee to include previous interactions as conversational history in the LLM prompt, enabling follow-up questions and contextual awareness.

<Note>
  Sessions require caching to be enabled. See the next sections and [Configuration Details](#configuration-details) below. If caching is disabled or unavailable, searches still work but without access to previous interactions.
</Note>

## How Sessions Work

Sessions are used during [search operations](/core-concepts/main-operations/search). When you call `cognee.search()` with a `session_id`:

1. **Retrieve context** – Cognee finds relevant graph elements for your query
2. **Load conversation history** – If caching is enabled, previous interactions for `(user_id, session_id)` are loaded
3. **Generate answer** – The LLM receives the query, graph context, and retrieved history
4. **Save interaction** – A new Q\&A entry is stored in the session cache

## Cache Adapters

Cognee supports two cache adapters for storing sessions. Redis is recommended for distributed or multi-process setups, while Filesystem can be used when you need a simple local cache without network dependencies. Both provide the same functionality; only the storage backend differs. Below are the configuration options for each adapter with additional details.

<Tabs>
  <Tab title="Redis">
    Add to your `.env` file:

    ```dotenv  theme={null}
    CACHING=true
    CACHE_BACKEND=redis
    CACHE_HOST=localhost
    CACHE_PORT=6379
    ```

    **Start Redis:**

    ```bash  theme={null}
    # Using Docker
    docker run -d -p 6379:6379 redis:latest

    # Or using local installation
    redis-server
    ```

    * Fast in-memory storage with built-in expiration
    * Supports shared locks for Kuzu (multi-process coordination)
    * Requires a running Redis instance and network connectivity
  </Tab>

  <Tab title="Filesystem">
    **Configuration:**

    Add to your `.env` file:

    ```dotenv  theme={null}
    CACHING=true
    CACHE_BACKEND=fs
    ```

    * Sessions are stored in `{DATA_ROOT_DIRECTORY}/.cognee_fs_cache/sessions_db`.
    * Stores session data on the local filesystem using `diskcache`
    * No network dependency
    * Does not provide shared locks for Kuzu
    * Not designed for multi-node coordination
  </Tab>
</Tabs>

<AccordionGroup>
  <Accordion title="Session Data Structure">
    Sessions store interactions as JSON entries in a list. Each entry contains:

    * **time**: ISO 8601 timestamp (UTC)
    * **question**: The user's query text
    * **context**: Summarized graph context used for the answer
    * **answer**: The LLM's response

    Sessions are keyed by: `agent_sessions:{user_id}:{session_id}`

    Each user can have multiple sessions, each maintaining its own cache of short-term information.
  </Accordion>

  <Accordion title="Configuration Details">
    **Environment Variables:**

    * `CACHING` (bool): Enable/disable caching (default: `false`)
    * `CACHE_BACKEND` (str): `"redis"` or `"fs"` (default: `"redis"` if `CACHING=true`)
    * `CACHE_HOST` (str): Redis hostname (default: `"localhost"`)
    * `CACHE_PORT` (int): Redis port (default: `6379`)
    * `CACHE_USERNAME` (str, optional): Redis username
    * `CACHE_PASSWORD` (str, optional): Redis password

    **TTL (Time to Live):**

    Sessions expire after a configurable TTL (default: 86400 seconds = 24 hours). Expired sessions are automatically cleaned up.
  </Accordion>

  <Accordion title="Adapter Comparison">
    | Feature          | Redis                   | Filesystem             |
    | ---------------- | ----------------------- | ---------------------- |
    | Storage          | In-memory (Redis)       | Local disk (diskcache) |
    | Performance      | Very fast               | Fast (local I/O)       |
    | Multi-process    | ✅ Supported             | ❌ Not supported        |
    | Shared locks     | ✅ Yes                   | ❌ No                   |
    | Network required | ✅ Yes                   | ❌ No                   |
    | Setup complexity | Medium                  | Low                    |
    | Best for         | Production, distributed | Development, local     |
  </Accordion>
</AccordionGroup>

<Columns cols={3}>
  <Card title="Search" icon="search" href="/core-concepts/main-operations/search">
    Learn how sessions integrate with search
  </Card>

  <Card title="Sessions Guide" icon="code" href="/guides/sessions">
    Practical examples with Redis and filesystem
  </Card>

  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/overview">
    Configure cache adapters
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt