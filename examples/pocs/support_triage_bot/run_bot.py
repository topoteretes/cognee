"""Support-Triage Bot — Entry Point.

Usage:
    python run_bot.py              # CLI mode (no external tokens needed)
    python run_bot.py --slack      # Slack mode (requires SLACK_BOT_TOKEN)
    python run_bot.py --seed       # CLI mode with pre-seeded demo threads

The bot watches a support channel, recalls similar past issues from
cognee memory, and suggests answers with citations to prior threads.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("support_triage_bot")


async def seed_demo_threads(adapter, ingest_handler):
    """Seed the bot with pre-built demo threads for testing."""
    from channel_adapters.cli_adapter import SEED_THREADS

    print("\n🌱 Seeding demo threads into cognee memory…\n")

    for thread_data in SEED_THREADS:
        result = await ingest_handler.handle(
            thread_id=thread_data["thread_id"],
            channel_id=thread_data["channel_id"],
            reporter=thread_data["reporter"],
            messages=thread_data["messages"],
            thread_url=thread_data["thread_url"],
        )
        status = result.get("status", "unknown")
        print(f"  ✅ Seeded {thread_data['thread_id']}: {thread_data['problem_summary'][:60]}… ({status})")

    print("\n🌱 Seeding complete! You can now query these threads.\n")


async def run_cli(seed: bool = False):
    """Run the bot in interactive CLI mode."""
    from channel_adapters.cli_adapter import CLIAdapter
    from config import BotConfig
    from handlers import ForgetHandler, IngestHandler, OptOutHandler, TriageHandler
    from memory_adapter import MemoryAdapter

    config = BotConfig()
    adapter = MemoryAdapter(config)
    cli = CLIAdapter()
    opt_out_list: set[str] = set()

    triage_handler = TriageHandler(adapter, config)
    ingest_handler = IngestHandler(adapter, config, opt_out_list)
    forget_handler = ForgetHandler(adapter)
    opt_out_handler = OptOutHandler(opt_out_list)

    print("=" * 60)
    print("  🤖 Support-Triage Bot (CLI Mode)")
    print("=" * 60)
    print()
    print("Commands:")
    print("  <any text>                → Triage: find similar past issues")
    print("  !resolve <id> <msg1|msg2> → Ingest a resolved thread")
    print("  !forget <id>              → Remove a thread from memory")
    print("  !optout                   → Opt out of future ingestion")
    print("  !status                   → Show stored thread mappings")
    print("  !quit / !exit             → Stop the bot")
    print()

    if seed:
        await seed_demo_threads(cli, ingest_handler)

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("support> ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("!quit", "!exit"):
            print("👋 Goodbye!")
            break

        if user_input.lower() == "!optout":
            result = opt_out_handler.handle("cli_user")
            print(f"✅ {result['message']}")
            continue

        if user_input.lower() == "!status":
            mappings = adapter._thread_id_to_data_id
            if mappings:
                print("\n📋 Stored thread mappings:")
                for tid, did in mappings.items():
                    print(f"  {tid} → {did}")
            else:
                print("\n📋 No thread mappings stored yet.")
            print()
            continue

        if user_input.startswith("!forget "):
            thread_id = user_input.split(" ", 1)[1].strip()
            result = await forget_handler.handle(thread_id)
            if result["status"] == "success":
                print(f"✅ Thread {thread_id} removed from memory.")
            else:
                print(f"❌ {result.get('message', 'Unknown error')}")
            continue

        if user_input.startswith("!resolve "):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                print("Usage: !resolve <thread_id> <msg1|msg2|msg3>")
                continue
            thread_id = parts[1]
            messages = parts[2].split("|")
            result = await ingest_handler.handle(
                thread_id=thread_id,
                channel_id="cli",
                reporter="cli_user",
                messages=[m.strip() for m in messages],
                thread_url=f"https://support.example.com/threads/{thread_id}",
            )
            status = result.get("status", "unknown")
            data_id = result.get("data_id", "N/A")
            print(f"✅ Thread {thread_id} ingested (status={status}, data_id={data_id})")
            continue

        # Default: triage the input as a new support query
        print("\n🔍 Searching for similar past issues…\n")
        result = await triage_handler.handle(user_input, "cli")
        reply = triage_handler.format_reply(result)
        await cli.send_reply("cli", "current", reply)


async def run_slack():
    """Run the bot in Slack mode."""
    from channel_adapters.slack_adapter import SlackAdapter
    from config import BotConfig
    from handlers import ForgetHandler, IngestHandler, OptOutHandler, TriageHandler
    from memory_adapter import MemoryAdapter

    config = BotConfig()

    if not config.slack_bot_token:
        print("❌ SLACK_BOT_TOKEN not set. Cannot start Slack mode.")
        sys.exit(1)
    if not config.slack_app_token:
        print("❌ SLACK_APP_TOKEN not set. Cannot start Slack mode.")
        sys.exit(1)

    adapter = MemoryAdapter(config)
    slack = SlackAdapter(
        bot_token=config.slack_bot_token,
        app_token=config.slack_app_token,
        signing_secret=config.slack_signing_secret,
    )
    opt_out_list: set[str] = set()

    triage_handler = TriageHandler(adapter, config)
    ingest_handler = IngestHandler(adapter, config, opt_out_list)
    forget_handler = ForgetHandler(adapter)
    opt_out_handler = OptOutHandler(opt_out_list)

    # Register Slack event listeners
    @slack.app.event("message")
    async def handle_message(event, say):
        """Triage new support messages."""
        text = event.get("text", "")
        channel = event.get("channel", "")
        user = event.get("user", "")
        thread_ts = event.get("thread_ts") or event.get("ts", "")

        if text.startswith("!"):
            return  # Skip commands — handled by command listeners

        result = await triage_handler.handle(text, channel)
        reply = triage_handler.format_reply(result)
        await slack.send_reply(channel, thread_ts, reply, ephemeral_user=user)

    @slack.app.event("reaction_added")
    async def handle_reaction(event, say):
        """Ingest resolved threads on ✅ reaction."""
        if event.get("reaction") != config.resolve_emoji:
            return

        item = event.get("item", {})
        channel = item.get("channel", "")
        thread_ts = item.get("ts", "")

        if not channel or not thread_ts:
            return

        messages = await slack.fetch_thread_messages(channel, thread_ts)
        thread_url = await slack.get_thread_permalink(channel, thread_ts)

        result = await ingest_handler.handle(
            thread_id=thread_ts,
            channel_id=channel,
            reporter=messages[0].user if messages else "unknown",
            messages=[m.text for m in messages],
            thread_url=thread_url,
        )
        logger.info("Slack ingest result: %s", result)

    @slack.app.command("/forget")
    async def handle_forget_command(ack, command):
        """Handle /forget slash command."""
        await ack()
        thread_id = command.get("text", "").strip()
        if not thread_id:
            return
        result = await forget_handler.handle(thread_id)
        logger.info("Slack forget result: %s", result)

    @slack.app.command("/optout")
    async def handle_optout_command(ack, command):
        """Handle /optout slash command."""
        await ack()
        user_id = command.get("user_id", "")
        result = opt_out_handler.handle(user_id)
        logger.info("Slack optout result: %s", result)

    print("🤖 Starting Support-Triage Bot (Slack mode)…")
    await slack.start()


def main():
    """Parse arguments and run the bot."""
    parser = argparse.ArgumentParser(
        description="Support-Triage Bot powered by cognee memory"
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Run in Slack mode (requires SLACK_BOT_TOKEN)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed demo threads into memory (CLI mode only)",
    )
    args = parser.parse_args()

    if args.slack:
        asyncio.run(run_slack())
    else:
        asyncio.run(run_cli(seed=args.seed))


if __name__ == "__main__":
    main()
