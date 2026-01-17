# Sessions

> Step-by-step guide to using sessions for conversational memory in Cognee

A minimal guide to enabling conversational memory with sessions. When you use the same `session_id` across searches, Cognee remembers previous questions and answers, enabling contextually aware follow-up questions.

## Before You Start

* Complete [Quickstart](../getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](../setup-configuration/llm-providers) configured
* Read [Sessions and Caching](../core-concepts/sessions-and-caching) for conceptual overview
* Configure your cache adapter before using sessions. See [Cache Adapters](../core-concepts/sessions-and-caching#cache-adapters) for Redis and Filesystem setup instructions.

## Code in Action

```python  theme={null}
import asyncio
import cognee
from cognee import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.modules.retrieval.utils.session_cache import set_session_user_context_variable

async def main():
    # Prepare knowledge base
    await cognee.add([
        "Alice moved to Paris in 2010. She works as a software engineer.",
        "Bob lives in New York. He is a data scientist.",
        "Alice and Bob met at a conference in 2015."
    ])
    await cognee.cognify()

    # Set user context (required for sessions)
    user = await get_default_user()
    await set_session_user_context_variable(user)

    # First search - starts a new session
    result1 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where does Alice live?",
        session_id="conversation_1"
    )
    print("First answer:", result1[0])

    # Follow-up search - uses conversation history
    result2 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What does she do for work?",
        session_id="conversation_1"  # Same session
    )
    print("Follow-up answer:", result2[0])
    # The LLM knows "she" refers to Alice from previous context

    # Different session - no memory of previous conversation
    result3 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What does she do for work?",
        session_id="conversation_2"  # New session
    )
    print("New session answer:", result3[0])
    # This won't know who "she" refers to

asyncio.run(main())
```

<Note>
  This example works with either Redis or Filesystem adapter. Configure your chosen adapter in the [Before you start](#before-you-start) section above.
</Note>

## What Just Happened

### Step 1: Prepare Knowledge Base

```python  theme={null}
await cognee.add([
    "Alice moved to Paris in 2010. She works as a software engineer.",
    "Bob lives in New York. He is a data scientist.",
    "Alice and Bob met at a conference in 2015."
])
await cognee.cognify()
```

Before you can search with sessions, you need to have data in your knowledge base. Use `cognee.add()` to ingest data and `cognee.cognify()` to build the knowledge graph.

### Step 2: Set User Context

```python  theme={null}
from cognee.modules.users.methods import get_default_user
from cognee.modules.retrieval.utils.session_cache import set_session_user_context_variable

user = await get_default_user()
await set_session_user_context_variable(user)
```

Sessions require a user context to associate conversation history with a specific user. This must be set before using `session_id` in searches.

### Step 3: Use Session ID in Searches

```python  theme={null}
result = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="Where does Alice live?",
    session_id="conversation_1"
)
```

The `session_id` parameter creates or continues a conversation thread. All searches with the same `session_id` share conversation history.

### Step 4: Follow-up Questions

```python  theme={null}
result = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="What does she do for work?",
    session_id="conversation_1"  # Same session
)
```

When you use the same `session_id`, Cognee automatically includes previous Q\&A turns in the LLM prompt, enabling contextual follow-up questions.

### Step 5: Multiple Sessions

```python  theme={null}
# Session 1
await cognee.search(query_text="Question 1", session_id="session_1")
await cognee.search(query_text="Follow-up", session_id="session_1")

# Session 2 (independent)
await cognee.search(query_text="Question 1", session_id="session_2")
```

Each `session_id` maintains its own conversation history. Sessions are isolated from each other.

## Advanced Usage

<Accordion title="Custom Session IDs">
  Use meaningful session IDs to organize conversations:

  ```python  theme={null}
  # User-specific sessions
  await cognee.search(query_text="...", session_id=f"user_{user_id}_chat")

  # Topic-specific sessions
  await cognee.search(query_text="...", session_id="project_planning")
  await cognee.search(query_text="...", session_id="bug_discussion")
  ```

  Session IDs are arbitrary strings—use whatever naming scheme fits your application.
</Accordion>

<Accordion title="Session Expiration">
  Sessions expire after 24 hours by default. To customize TTL, configure it in your cache adapter settings. Expired sessions are automatically cleaned up and won't affect new searches.
</Accordion>

<Accordion title="Disabling Sessions">
  If caching is disabled or unavailable, searches work normally but without conversational memory:

  ```dotenv  theme={null}
  CACHING=false
  ```

  Or simply omit `session_id` from search calls. The system gracefully handles missing cache backends.
</Accordion>

<Columns cols={3}>
  <Card title="Sessions and Caching" icon="brain" href="/core-concepts/sessions-and-caching">
    Understand how sessions work conceptually
  </Card>

  <Card title="Search Basics" icon="search" href="/guides/search-basics">
    Learn about search parameters and types
  </Card>

  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/overview">
    Configure cache adapters and providers
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt