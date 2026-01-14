# Search Basics

> Step-by-step guide to running your first Cognee search and understanding core parameters

A minimal guide to using `cognee.search()` to ask questions against your processed datasets. This guide shows the basic call and what each parameter does so you know which knob to turn.

**Before you start:**

* Complete [Quickstart](../getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](../setup-configuration/llm-providers) configured for LLM-backed search types
* Run `cognee.cognify(...)` to build the graph before searching
* Keep at least one dataset with `read` permission for the user running the search

## Code in Action

```python  theme={null}
import asyncio
import cognee

async def main():
    # Make sure you've already run cognee.cognify(...) so the graph has content
    answers = await cognee.search(
        query_text="What are the main themes in my data?"
    )
    for answer in answers:
        print(answer)

asyncio.run(main())
```

<Note>
  `SearchType.GRAPH_COMPLETION` is the default, so you get an LLM-backed answer plus supporting context as soon as you have data in your graph.
</Note>

## What Just Happened

The search call uses the default `SearchType.GRAPH_COMPLETION` mode to provide LLM-backed answers with supporting context from your knowledge graph. The results are returned as a list that you can iterate through and process as needed.

## Parameters Reference

Most examples below assume you are inside an async function. Import helpers when you need them:

```python  theme={null}
from cognee import SearchType
from cognee.modules.engine.models.node_set import NodeSet
```

<AccordionGroup>
  <Accordion title="Core Parameters" defaultOpen={true}>
    * **`query_text`** (str, required): The question or phrase you want answered.
      ```python  theme={null}
      answers = await cognee.search(query_text="Who owns the rollout plan?")
      ```
    * **`query_type`** (SearchType, optional, default: `SearchType.GRAPH_COMPLETION`): Switch search modes without changing your code flow. See [Search Types](../core-concepts/main-operations/search) for the complete list.
      ```python  theme={null}
      await cognee.search(
          query_text="List coding guidelines",
          query_type=SearchType.CODING_RULES,
      )
      ```
    * **`top_k`** (int, optional, default: 10): Cap how many ranked results you want back.
      ```python  theme={null}
      await cognee.search(query_text="Summaries please", top_k=3)
      ```
  </Accordion>

  <Accordion title="Prompt & Generation Parameters">
    * **`system_prompt_path`** (str, optional, default: `"answer_simple_question.txt"`): Point to a prompt file packaged with your project.
      ```python  theme={null}
      await cognee.search(
          query_text="Explain the roadmap in bullet points",
          system_prompt_path="prompts/bullets.txt",
      )
      ```
    * **`system_prompt`** (Optional\[str]): Inline override for experiments or dynamically generated prompts.
      ```python  theme={null}
      await cognee.search(
          query_text="Give me a confident answer",
          system_prompt="Answer succinctly and state confidence at the end.",
      )
      ```
    * **`only_context`** (bool, optional, default: False): Skip LLM generation and just fetch supporting context chunks.
      ```python  theme={null}
      context = await cognee.search(
          query_text="What did we promise the client?",
          only_context=True,
      )
      ```
    * **`use_combined_context`** (bool, optional, default: False): Collapse results into a single combined response when you query multiple datasets.
      ```python  theme={null}
      combined = await cognee.search(
          query_text="Quarterly financial highlights",
          datasets=["finance_q1", "finance_q2"],
          use_combined_context=True,
      )
      ```

    <Info>
      `use_combined_context` should only be set when `ENABLE_BACKEND_ACCESS_CONTROL` is turned on. When access control is disabled, this parameter has no meaningful effect on dataset scoping.
    </Info>
  </Accordion>

  <Accordion title="Node Sets & Filtering Parameters">
    These options filter the graph down to the node sets you care about. In most workflows you set **both**: keep `node_type=NodeSet` and pass one or more set names in `node_name`—the same labels you used when calling `cognee.add(..., node_set=[...])`.

    * **`node_type`** (Optional\[Type], optional, default: `NodeSet`): Controls which graph model to search. Leave this as `NodeSet` unless you’ve built a custom node model.
    * **`node_name`** (Optional\[List\[str]]): Names of the node sets to include. Cognee treats each string as a logical bucket of memories.
      ```python  theme={null}
      await cognee.search(
          query_text="What discounts did TechSupply offer?",
          node_type=NodeSet,
          node_name=["vendor_conversations"],
      )
      ```
      ```python  theme={null}
      await cognee.search(
          query_text="Summarize procurement rules",
          node_type=NodeSet,
          node_name=["procurement_policies", "purchase_history"],
      )
      ```
  </Accordion>

  <Accordion title="Interaction & History Parameters">
    * **`session_id`** (Optional\[str]): Maintain conversation history across searches. When you use the same `session_id`, Cognee includes previous interactions in the LLM prompt, enabling contextual follow-up questions.
      ```python  theme={null}
      await cognee.search(
          query_text="Where does Alice live?",
          session_id="conversation_1"
      )
      # Later, same session remembers previous context
      await cognee.search(
          query_text="What does she do for work?",
          session_id="conversation_1"  # "she" refers to Alice
      )
      ```
      See [Sessions Guide](/guides/sessions) for complete examples.
    * **`save_interaction`** (bool, optional, default: False): Persist the Q\&A as a graph interaction for auditing or later review.
      ```python  theme={null}
      await cognee.search(
          query_text="Draft the release note",
          save_interaction=True,
      )
      ```
    * **`last_k`** (Optional\[int], optional, default: 1): When using `SearchType.FEEDBACK`, choose how many recent interactions to update with your feedback.
      ```python  theme={null}
      await cognee.search(
          query_text="Please improve the last answer",
          query_type=SearchType.FEEDBACK,
          last_k=3,
      )
      ```
  </Accordion>

  <Accordion title="Datasets & Users">
    * **`datasets`** (Optional\[Union\[list\[str], str]]): Limit search to dataset names you already know.
      ```python  theme={null}
      await cognee.search(
          query_text="Key risks",
          datasets=["risk_register", "exec_summary"],
      )
      ```
    * **`dataset_ids`** (Optional\[Union\[list\[UUID], UUID]]): Same as `datasets`, but with explicit UUIDs when names collide.
      ```python  theme={null}
      from uuid import UUID
      await cognee.search(
          query_text="Customer feedback",
          dataset_ids=[UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")],
      )
      ```
    * **`user`** (Optional\[User]): Provide a user object when running multi-tenant flows or background jobs.

      ```python  theme={null}
      from cognee.modules.users.methods import get_user
      user = await get_user(user_id)
      await cognee.search(query_text="Team OKRs", user=user)
      ```

      **When** `ENABLE_BACKEND_ACCESS_CONTROL=true`:

      * **Result shape**: Searches run only on datasets the user can access and return either:
        * **Per dataset**: list of `{dataset_name, dataset_id, search_result}`
        * **Combined**: single `CombinedSearchResult` with merged snippets (`use_combined_context=True`)

      * If no `user` is given, `get_default_user()` is used (created if missing); errors only if this user lacks dataset permissions.

      * If `datasets` is not set, all datasets readable by the user are searched; errors if none are accessible or if requested datasets are forbidden.

      <Warning>
        `PermissionDeniedError` will be raised unless you search with the same user that added the data or grant access to the default user.
      </Warning>

      **When** `ENABLE_BACKEND_ACCESS_CONTROL=false`

      * Dataset filters (`datasets`, `dataset_ids`) are ignored — everything is searched.
      * Results normally come back as a plain list (`["answer1", "answer2"]`).
      * Setting `use_combined_context=True` here just wraps the same results in a `CombinedSearchResult` without changing them.
  </Accordion>
</AccordionGroup>

<Columns cols={2}>
  <Card title="Custom Prompts" icon="text-wrap" href="/guides/custom-prompts">
    Learn about custom prompts for tailored answers
  </Card>

  <Card title="Permission Snippets" icon="shield" href="/guides/permission-snippets">
    Multi-tenant deployment patterns
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore all search types and parameters
  </Card>

  <Card title="Sessions" icon="message-square" href="/guides/sessions">
    Enable conversational memory with sessions
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt