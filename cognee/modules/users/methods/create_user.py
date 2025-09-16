from uuid import UUID, uuid4
from fastapi_users.exceptions import UserAlreadyExists
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.notebooks.methods import create_notebook
from cognee.modules.notebooks.models.Notebook import NotebookCell
from cognee.modules.users.exceptions import TenantNotFoundError
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.users.models.User import UserCreate
from cognee.modules.users.models.Tenant import Tenant

from sqlalchemy import select
from typing import Optional


async def _create_tutorial_notebook(user_id: UUID, session: AsyncSession) -> None:
    """
    Create the default tutorial notebook for new users.
    Tutorial notebook: https://github.com/topoteretes/cognee/blob/notebook_tutorial/notebooks/tutorial.ipynb
    """
    cells = [
        NotebookCell(
            id=uuid4(),
            name="Tutorial Introduction",
            content="# Using Cognee with Python Development Data\n\nUnite authoritative Python practice (Guido van Rossum's own contributions!), normative guidance (Zen/PEP 8), and your lived context (rules + conversations) into one *AI memory* that produces answers that are relevant, explainable, and consistent.",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="What You'll Learn",
            content="## What You'll Learn\n\nIn this comprehensive tutorial, you'll discover how to transform scattered development data into an intelligent knowledge system that enhances your coding workflow. By the end, you'll have:\n\n- **Connected disparate data sources** (Guido's CPython contributions, mypy development, PEP discussions, your Python projects) into a unified AI memory graph\n- **Built a memory layer** that understands Python design philosophy, best practice coding patterns, and your preferences and experience\n- **Learn how to use intelligent search capabilities** that combine the diverse context\n- **Integrated everything with your coding environment** through MCP (Model Context Protocol)\n\nThis tutorial demonstrates the power of **knowledge graphs** and **retrieval-augmented generation (RAG)** for software development, showing you how to build systems that learn from Python's creator and improve your own Python development.",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Core Cognee Operations",
            content="## Cognee and its core operations\n\nBefore we dive in, let's understand the core Cognee operations we'll be working with:\n\n- **`cognee.add()`** - Ingests raw data (files, text, APIs) into the system\n- **`cognee.cognify()`** - Processes and structures data into a knowledge graph using AI\n- **`cognee.search()`** - Queries the knowledge graph with natural language or Cypher\n- **`cognee.memify()`** - Cognee's \"secret sauce\" that infers implicit connections and rules from your data",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Tutorial Data Sources",
            content='## Data used in this tutorial\n\nCognee can ingest many types of sources. In this tutorial, we use a small, concrete set of files that cover different perspectives:\n\n- **`guido_contributions.json` — Authoritative exemplars.** Real PRs and commits from Guido van Rossum (mypy, CPython). These show how Python\'s creator solved problems and provide concrete anchors for patterns.\n- **`pep_style_guide.md` — Norms.** Encodes community style and typing conventions (PEP 8 and related). Ensures that search results and inferred rules align with widely accepted standards.\n- **`zen_principles.md` — Philosophy.** The Zen of Python. Grounds design trade‑offs (simplicity, explicitness, readability) beyond syntax or mechanics.\n- **`my_developer_rules.md` — Local constraints.** Your house rules, conventions, and project‑specific requirements (scope, privacy, Spec.md). Keeps recommendations relevant to your actual workflow.\n- **`copilot_conversations.json` — Personal history.** Transcripts of real assistant conversations, including your questions, code snippets, and discussion topics. Captures "how you code" and connects it to "how Guido codes."',
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Setup: Prerequisites",
            content='# Preliminaries\n\nCognee relies heavily on async functions.\nWe need `nest_asyncio` so `await` works in this notebook.\n\nTo strike the balance between speed, cost, and quality, we recommend using OpenAI\'s `4o-mini` model; make sure your `.env` file contains this line:\n\n```\nLLM_MODEL="gpt-4o-mini"\n```',
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Setup: Import Check",
            content="import nest_asyncio\nnest_asyncio.apply()\n\nimport cognee\nimport os\nfrom pathlib import Path\n\nprint('🔍 Quick Cognee Import Check')\nprint('=' * 30)\nprint(f'📍 Cognee location: {cognee.__file__}')\nprint(f'📁 Package directory: {os.path.dirname(cognee.__file__)}')\n\n# Check if it's local or installed\ncurrent_dir = Path.cwd()\ncognee_path = Path(cognee.__file__)\nif current_dir in cognee_path.parents:\n    print('🏠 Status: LOCAL DEVELOPMENT VERSION')\nelse:\n    print('📦 Status: INSTALLED PACKAGE')",
            type="code",
        ),
        NotebookCell(
            id=uuid4(),
            name="Data Ingestion",
            content="## Step 1: Data Ingestion\n\nNow let's add our tutorial data sources to Cognee. In a real scenario, you would have your own data files.",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Add Tutorial Data",
            content='# Add sample data - replace with your actual file paths\n# In this tutorial, we\'re simulating the data sources mentioned above\n\n# Example: Adding Guido\'s contributions\n# await cognee.add("path/to/guido_contributions.json")\n\n# Example: Adding PEP style guide\n# await cognee.add("path/to/pep_style_guide.md")\n\n# Example: Adding Zen principles\n# await cognee.add("path/to/zen_principles.md")\n\n# Example: Adding your developer rules\n# await cognee.add("path/to/my_developer_rules.md")\n\n# Example: Adding conversation history\n# await cognee.add("path/to/copilot_conversations.json")\n\nprint("📁 Sample data sources prepared for ingestion")\nprint("Replace the commented lines above with your actual file paths")',
            type="code",
        ),
        NotebookCell(
            id=uuid4(),
            name="Knowledge Graph Creation",
            content="## Step 2: Knowledge Graph Creation (Cognify)\n\nAfter adding data, we use `cognee.cognify()` to process and structure it into a knowledge graph.",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Process Data into Graph",
            content='# Process the added data into a knowledge graph\n# This step analyzes the content and creates relationships\n\n# Example cognification with temporal awareness\n# result = await cognee.cognify(temporal_cognify=True)\n\nprint("🧠 Knowledge graph creation completed")\nprint("Your data is now structured and ready for intelligent querying")',
            type="code",
        ),
        NotebookCell(
            id=uuid4(),
            name="Intelligent Search",
            content="## Step 3: Intelligent Search\n\nNow you can query your knowledge graph using natural language. Cognee supports various search types including temporal and graph completion searches.",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Example Search Queries",
            content='# Example searches you can perform\n\n# Basic search\n# result = await cognee.search(\n#     query_text="What are Python best practices according to the data?"\n# )\n\n# Temporal search\n# result = await cognee.search(\n#     query_text="What can we learn from recent contributions?",\n#     query_type=cognee.SearchType.TEMPORAL\n# )\n\n# Graph completion search\n# result = await cognee.search(\n#     query_type=cognee.SearchType.GRAPH_COMPLETION,\n#     query_text="What is the most zen thing about Python?",\n#     save_interaction=True  # Enable feedback loops\n# )\n\nprint("🔍 Search examples ready - uncomment and run the queries above")',
            type="code",
        ),
        NotebookCell(
            id=uuid4(),
            name="Feedback System",
            content="## Step 4: Feedback and Learning\n\nCognee can learn from your feedback to improve future responses. When you enable `save_interaction=True`, you can provide feedback that gets incorporated into the knowledge graph.",
            type="markdown",
        ),
        NotebookCell(
            id=uuid4(),
            name="Provide Feedback",
            content='# After running a search with save_interaction=True, you can provide feedback\n\n# Example feedback\n# feedback = await cognee.search(\n#     query_type=cognee.SearchType.FEEDBACK,\n#     query_text="Last result was useful, I like code that complies with best practices.",\n#     last_k=1\n# )\n\nprint("💡 Feedback system ready - provide feedback to improve future results")',
            type="code",
        ),
        NotebookCell(
            id=uuid4(),
            name="Next Steps",
            content="## Next Steps\n\n🎉 **Congratulations!** You've learned how to:\n\n1. **Ingest diverse data sources** into Cognee\n2. **Create intelligent knowledge graphs** with `cognify()`\n3. **Query your data** using natural language search\n4. **Provide feedback** for continuous learning\n\n### What to try next:\n\n- Add your own Python projects and documentation\n- Experiment with different search types and queries\n- Integrate with your development workflow through MCP\n- Explore temporal queries to understand code evolution\n- Build custom ontologies for domain-specific knowledge\n\n### Resources:\n\n- [Cognee Documentation](https://docs.cognee.ai)\n- [GitHub Repository](https://github.com/topoteretes/cognee)\n- [Community Discussions](https://github.com/topoteretes/cognee/discussions)",
            type="markdown",
        ),
    ]

    await create_notebook(
        user_id=user_id,
        notebook_name="Python Development with Cognee Tutorial 🧠",
        cells=cells,
        deletable=False,
        session=session,
    )


async def create_user(
    email: str,
    password: str,
    tenant_id: Optional[str] = None,
    is_superuser: bool = False,
    is_active: bool = True,
    is_verified: bool = False,
    auto_login: bool = False,
):
    try:
        relational_engine = get_relational_engine()

        async with relational_engine.get_async_session() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    if tenant_id:
                        # Check if the tenant already exists
                        result = await session.execute(select(Tenant).where(Tenant.id == tenant_id))
                        tenant = result.scalars().first()
                        if not tenant:
                            raise TenantNotFoundError

                        user = await user_manager.create(
                            UserCreate(
                                email=email,
                                password=password,
                                tenant_id=tenant.id,
                                is_superuser=is_superuser,
                                is_active=is_active,
                                is_verified=is_verified,
                            )
                        )
                    else:
                        user = await user_manager.create(
                            UserCreate(
                                email=email,
                                password=password,
                                is_superuser=is_superuser,
                                is_active=is_active,
                                is_verified=is_verified,
                            )
                        )

                    if auto_login:
                        await session.refresh(user)

                    await _create_tutorial_notebook(user.id, session)

                    return user
    except UserAlreadyExists as error:
        print(f"User {email} already exists")
        raise error
