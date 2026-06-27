import asyncio

from cognee.tasks.ingestion.save_data_item_to_storage import (
    IngestionError,
    save_data_item_to_storage,
    settings,
)


async def test():
    # 1. Test when local files are ALLOWED (Should succeed and return a file:// URI)
    settings.accept_local_file_path = True
    try:
        result = await save_data_item_to_storage("dummy.txt")
        print(f"✅ Test 1 Passed: Allowed path returned -> {result}")
    except Exception as e:
        print(f"❌ Test 1 Failed: Unexpected error -> {e}")

    # 2. Test when local files are DISABLED (Should raise IngestionError)
    settings.accept_local_file_path = False
    try:
        await save_data_item_to_storage("dummy.txt")
        print("❌ Test 2 Failed: Silently bypassed! No error was raised.")
    except IngestionError as e:
        print(f"✅ Test 2 Passed: Correctly blocked relative path! Error: {e}")
    except Exception as e:
        print(f"❌ Test 2 Failed: Raised the wrong exception type -> {type(e).__name__}: {e}")


# Run the async test
asyncio.run(test())
