# Chat Memory Adapter examples

Reference bots built on
[`cognee.integrations.chat_memory`](../../../cognee/integrations/chat_memory/).

| file | needs keys? | what it shows |
| --- | --- | --- |
| [`console_bot.py`](./console_bot.py) | no | The full contract (consent, ingest, answer-with-citations, forget-me) end to end on the in-memory backend. Run it to see the adapter work with zero setup. |
| [`telegram_bot.py`](./telegram_bot.py) | yes | A ~100-line real Telegram bot on cognee. The whole platform layer is one `conversation_of()` function; everything else is inherited from the core. |

## Run the console bot (no keys)

```bash
python examples/integrations/chat_memory/console_bot.py
```

## Run the Telegram bot

```bash
pip install "cognee[anthropic]" python-telegram-bot
export LLM_API_KEY=...            # cognee graph memory
export TELEGRAM_BOT_TOKEN=...     # from @BotFather
python examples/integrations/chat_memory/telegram_bot.py
```
