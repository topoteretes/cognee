---
name: summarize
description: >
  Summarize documents, articles, or text into concise key points.
  Use when the user asks to "summarize this", "give me the key points",
  "TL;DR", "what are the main takeaways", or wants a shorter version
  of any text content.
---

# Text Summarization

This skill produces concise summaries of documents, articles, and text blocks.

## When to Activate

- User provides a document and asks for a summary
- User pastes text and wants key takeaways
- User asks for a TL;DR of any content

## Process

1. Read the full input text
2. Identify the main topics and key arguments
3. Produce a structured summary with:
   - One-line TL;DR
   - 3-5 bullet points of key findings
   - Notable quotes or data points (if any)

## Output Format

```
## TL;DR
[One sentence summary]

## Key Points
- Point 1
- Point 2
- Point 3

## Notable Details
- [Any specific data, quotes, or references worth preserving]
```

## Guidelines

1. Preserve factual accuracy -- never add claims not in the source
2. Keep summaries under 20% of the original length
3. Use the source's terminology, not synonyms
4. Flag if the source contains conflicting information
