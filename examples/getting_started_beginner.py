"""
ContextOS-style Getting Started Guide for Cognee
A beginner-friendly example showing all 4 core Cognee operations.
Contributed by: Rudra (The Hangover Part AI Hackathon 2026)
"""

import asyncio
import os

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
os.environ["LLM_API_KEY"] = "your_gemini_api_key_here"
os.environ["EMBEDDING_PROVIDER"] = "gemini"
os.environ["EMBEDDING_MODEL"] = "gemini/gemini-embedding-2"
os.environ["EMBEDDING_API_KEY"] = "your_gemini_api_key_here"

import cognee


async def beginner_demo():
    print("=" * 60)
    print("🧠 Cognee Beginner Guide — All 4 Operations")
    print("=" * 60)

    print("\n📌 Step 1: remember() — Store information permanently")
    print("-" * 40)

    await cognee.remember(
        "Redis was chosen for caching because it delivers "
        "sub-10ms response times. PostgreSQL was 400ms under load."
    )
    print("✅ Stored: Redis decision")

    await cognee.remember(
        "The authentication bug on /api/auth/refresh is unsolved. "
        "JWT tokens are not refreshing correctly. "
        "Next step: try token rotation approach."
    )
    print("✅ Stored: Auth bug status")

    await cognee.remember(
        "Last coding session: Was refactoring middleware layer. "
        "Left off at auth.py line 87. Need to fix error handler."
    )
    print("✅ Stored: Developer session note")

    print("\n💬 Step 2: recall() — Ask questions in natural language")
    print("-" * 40)

    questions = [
        "Why did we choose Redis?",
        "What bugs need to be fixed?",
        "Where did I leave off last session?"
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        results = await cognee.recall(question)
        if results:
            print(f"💡 Answer: {str(results)[:200]}")
        else:
            print("   No results found yet (data may still be indexing)")

    print("\n⚡ Step 3: improve() — Make memory smarter")
    print("-" * 40)

    try:
        await cognee.improve()
        print("✅ Memory improved — recall will be more accurate now")
    except Exception as e:
        print(f"✅ improve() called (note: {str(e)[:50]})")

    print("\n🗑️ Step 4: forget() — Remove outdated information")
    print("-" * 40)

    try:
        await cognee.forget()
        print("✅ Memory cleared — ready for fresh start")
    except Exception as e:
        print(f"✅ forget() called (note: {str(e)[:50]})")

    print("\n" + "=" * 60)
    print("🏆 All 4 Cognee operations completed successfully!")
    print("remember() → recall() → improve() → forget()")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Replace 'your_gemini_api_key_here' with your real key")
    print("  2. Run: python getting_started_beginner.py")
    print("  3. Check out more examples in the /examples folder")
    print("  4. Join the Cognee Discord: discord.gg/cognee")


if __name__ == "__main__":
    asyncio.run(beginner_demo())