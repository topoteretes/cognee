---
name: data-extraction
description: >
  Extract structured data from unstructured text, PDFs, or web pages.
  Use when the user asks to "extract data from", "pull out the fields",
  "parse this into a table", "get the entities from", or needs to
  convert free-form text into structured JSON, CSV, or database records.
---

# Data Extraction

This skill extracts structured data from unstructured sources.
let;
## When to Activate

- User provides text and wants specific fields extracted
- User has a PDF or document and needs tabular data from it
- User wants to convert free-form text into JSON or CSV

## Process

1. Identify the source format (text, PDF, HTML, etc.)
2. Determine the target schema (from user instructions or infer from content)
3. Extract entities and fields
4. Validate extracted data for completeness
5. Return in the requested format

## Output Format

Default to JSON unless the user requests otherwise:

```json
{
  "extracted_records": [
    {"field1": "value1", "field2": "value2"}
  ],
  "confidence": 0.95,
  "warnings": ["any ambiguous extractions noted here"]
}
```

## Guidelines

1. Always include a confidence score for the extraction
2. Flag ambiguous or uncertain extractions in warnings
3. Preserve the original text for fields that are quoted verbatim
4. If the schema is not specified, propose one before extracting
