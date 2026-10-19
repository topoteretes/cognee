# Twilio Data-Source Connector

Ingests SMS/MMS messages, voice call logs, and call audio transcriptions from [Twilio](https://www.twilio.com/) into Cognee memory and entity knowledge graphs.

## Features

- **Multi-Stream Ingestion:** Ingests SMS/MMS messages, voice call records, and call audio transcriptions.
- **Relational Phone Entity Graphs:** Automatically groups interactions by customer phone numbers (in E.164 format) into Cognee node sets (`phone:+1555...`), connecting communications into unified customer memory.
- **Incremental Sync:** Tracks timestamps (`DateSent>=<floor>`, `StartTime>=<floor>`) so subsequent runs only ingest new interactions.
- **Privacy & PII Compliance:** Optional `mask_phone_numbers=True` redacts phone numbers (preserving country code and last 4 digits) for GDPR and HIPAA compliance.
- **Rate Limit & Retry Handling:** Automatically handles Twilio HTTP 429 and server errors with exponential backoff.
- **Deduplication & Merging:** Uses `write_disposition="merge"` with deterministic document IDs (`twilio:msg:<sid>`, `twilio:call:<sid>`, `twilio:transcript:<sid>`).

## Setup & Credentials

To use the Twilio connector, retrieve your Twilio credentials from the [Twilio Console](https://console.twilio.com/):
- **Account SID:** Starts with `AC...`
- **Auth Token:** Your primary API secret, or a custom API Key (`SK...`) and Secret.

## Quickstart Example

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import twilio_source

async def main():
    # Configure Cognee memory ingestion from Twilio
    source = twilio_source(
        account_sid="AC_YOUR_TWILIO_ACCOUNT_SID",
        auth_token="your_twilio_auth_token",
        include_messages=True,
        include_calls=True,
        include_transcriptions=True,
        mask_phone_numbers=False,
    )

    # Ingest into Cognee memory
    await cognee.remember(
        source,
        dataset_name="customer_telecom",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # Query customer telecommunications memory
    results = await cognee.recall("What did customer +1-555-0199 text us about their appointment?")
    print("Recall Results:", results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `account_sid` | `str` | Required | Twilio Account SID (`AC...`). |
| `auth_token` | `str` | Required | Twilio Auth Token or API Secret. |
| `api_key` | `str` | `None` | Optional API Key SID (`SK...`) if using API Key authentication. |
| `include_messages` | `bool` | `True` | Whether to ingest SMS and MMS text messages. |
| `include_calls` | `bool` | `True` | Whether to ingest voice call logs and metadata. |
| `include_transcriptions` | `bool` | `True` | Whether to ingest voice call audio transcriptions. |
| `mask_phone_numbers` | `bool` | `False` | Redacts middle digits of phone numbers for PII compliance. |
| `page_size` | `int` | `50` | Pagination page size (max 100). |
| `max_requests` | `int` | `1000` | Maximum API requests permitted in a single sync execution. |

## Graph Node Set Structure

The connector tags generated documents with structured node sets:
- `twilio`: Master namespace.
- `twilio:sms`: SMS and MMS messages.
- `twilio:call`: Voice call logs.
- `twilio:transcript`: Call recordings audio transcriptions.
- `phone:<e164_number>`: Dynamic customer phone identity node linking all calls and texts with that number.
