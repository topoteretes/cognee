import asyncio
import traceback

from cognee.infrastructure.llm.LLMGateway import LLMGateway

async def main():
    try:
        print("Starting Gemini smoke test (minimal call)...")
        coro = LLMGateway.acreate_structured_output(
            text_input="Say hello in one short sentence.",
            system_prompt="You are a helpful assistant.",
            response_model=str,
        )
        res = await coro
        print("---SUCCESS---")
        # print a short snippet of the response
        try:
            print(str(res)[:1000])
        except Exception:
            print(repr(res))
    except Exception as e:
        print("---ERROR---")
        traceback.print_exc()
        print("Exception:", e)

if __name__ == '__main__':
    asyncio.run(main())
