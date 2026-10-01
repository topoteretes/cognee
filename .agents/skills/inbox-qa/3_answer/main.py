"""Step 3: answer a question from everything the earlier steps remembered.

Facts come from the newest email and your Granola meetings; when the question asks for a
reply, a few of your own sent emails are passed in so the draft sounds like you.

Run alone: uv run python .agents/skills/inbox-qa/3_answer/main.py "Your question"
"""

import asyncio
import sys

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "inbox_qa_skill"  # the same in every step
DEFAULT_QUESTION = "Draft my reply to the newest email, using what my meetings say about it."
STYLE_PROMPT = """Answer from the context. Never invent a date, price or promise.
When asked to write or draft a reply, write it as the user would: match the greeting,
length, tone and sign-off of the user's own emails given below. Write in the language of
the email being answered, not the language of the examples."""


async def main(question: str) -> None:
    # Name the inbox email explicitly: with sent mail in the graph too, "the newest email"
    # alone can match one of your own threads instead.
    inbox = await cognee.recall(
        "The newest email in my inbox",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        node_name=["email"],
        top_k=1,
    )
    own_emails = await cognee.recall(
        "Emails I wrote to a person",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        node_name=["sent"],
        top_k=3,
    )
    prompt = question
    if inbox:
        prompt += f"\n\nThe newest email in my inbox:\n{inbox[0].text}"
    if own_emails:
        examples = "\n---\n".join(str(chunk.text) for chunk in own_emails)
        prompt += f"\n\nExamples of my own emails, for style only:\n{examples}"
    results = await cognee.recall(
        prompt,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        system_prompt=STYLE_PROMPT,
    )
    answer = str(results[0].text) if results else "Nothing found. Run the ingest steps first."
    print(f"[answer] Q: {question}\n[answer] A: {answer}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
