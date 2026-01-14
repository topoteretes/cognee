# Feedback System

> Step-by-step guide to using feedback to improve Cognee's knowledge graphs

This guide shows you how to use Cognee's feedback system to improve search results and knowledge graph quality.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Read [Search](/core-concepts/main-operations/search) to understand search types
* Ensure you have [LLM Providers](/setup-configuration/llm-providers) configured for feedback processing

## Example: Basic Feedback Loop

This example shows how to provide feedback to improve future search results.

### Step 1: Perform Search with Interaction Saving

```python  theme={null}
import cognee
from cognee import SearchType

# Search with interaction saving enabled
results = await cognee.search(
    query_text="What are the main themes in my data?",
    query_type=SearchType.GRAPH_COMPLETION,
    save_interaction=True  # Required for feedback
)

print("Search results:", results)
```

### Step 2: Provide Positive Feedback

```python  theme={null}
# Provide positive feedback
await cognee.search(
    query_text="Excellent analysis, very comprehensive and accurate!",
    query_type=SearchType.FEEDBACK,
    last_k=1  # Apply to last interaction
)

print("✅ Positive feedback applied")
```

### Step 3: Provide Negative Feedback

```python  theme={null}
# Provide constructive negative feedback
await cognee.search(
    query_text="This answer missed the key technical details I needed",
    query_type=SearchType.FEEDBACK,
    last_k=1
)

print("✅ Negative feedback applied")
```

**Result:** Feedback scores are applied to knowledge graph relationships to improve future results.

## Example: Batch Feedback Collection

This example shows how to collect feedback on multiple recent interactions.

### Step 1: Perform Multiple Searches

```python  theme={null}
# Perform several searches
queries = [
    "What are the technical requirements?",
    "Summarize the project timeline",
    "Explain the architecture decisions"
]

for query in queries:
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        save_interaction=True
    )
    print(f"Results for '{query}': {results}")
```

### Step 2: Provide Batch Feedback

```python  theme={null}
# Provide feedback on multiple recent interactions
await cognee.search(
    query_text="The last few searches have been much more accurate and helpful",
    query_type=SearchType.FEEDBACK,
    last_k=3  # Apply to last 3 interactions
)

print("✅ Batch feedback applied to recent interactions")
```

**Result:** Multiple interactions are improved based on your feedback.

## Example: Application Integration

This example shows how to integrate feedback collection in your application.

### Step 1: Create Search Function with Feedback

```python  theme={null}
async def search_with_feedback(query: str, user_feedback: str = None):
    # Perform search
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        save_interaction=True
    )
    
    # If user provides feedback, apply it
    if user_feedback:
        await cognee.search(
            query_text=user_feedback,
            query_type=SearchType.FEEDBACK,
            last_k=1
        )
        print("✅ Feedback collected and applied")
    
    return results
```

### Step 2: Use in Your Application

```python  theme={null}
# Search with immediate feedback
results = await search_with_feedback(
    "What are the security considerations?",
    "Great answer, very detailed and practical"
)

# Search without feedback
results = await search_with_feedback("What is the deployment process?")
```

**Result:** Integrated feedback collection in your application workflow.

## Common Issues

**Feedback not working:**

* Ensure `save_interaction=True` in your search calls
* Check that you have recent interactions to apply feedback to
* Verify you're using `SearchType.FEEDBACK` for feedback calls

**No improvement in results:**

* Provide more specific feedback text
* Give feedback soon after receiving results
* Use positive feedback to reinforce good results

**Performance concerns:**

* Feedback requires LLM processing for sentiment analysis
* Consider batching multiple feedback calls
* Monitor LLM API quotas and rate limits

**Integration challenges:**

* Start with simple feedback collection
* Gradually add more sophisticated feedback patterns
* Test feedback effectiveness over time

<Columns cols={2}>
  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore feedback API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt