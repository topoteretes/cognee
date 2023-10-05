import os
import openai
from deepeval.metrics.factual_consistency import assert_factual_consistency
import dotenv
dotenv.load_dotenv()
from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
openai.api_key = os.getenv("OPENAI_API_KEY", "")
from deepeval.metrics.overall_score import assert_overall_score
import json
from deepeval.metrics.overall_score import OverallScoreMetric

# Needs to pass a QA test set

# Needs to separately fetch QA test set from DB

# Needs to have a function to run the tests that contain test set and store results in the DB




async def main():
    async def generate_context(query: str='bla', context:str=None):
        memory = Memory(user_id="TestUser")

        await memory.async_init()


        memory_loaded = await memory._fetch_semantic_memory(observation=query, params=None)

        if memory_loaded:
            return memory_loaded["data"]["Get"]["SEMANTICMEMORY"][0]["text"]
        else:
            params = {
                "version": "1.0",
                "agreement_id": "AG123456",
                "privacy_policy": "https://example.com/privacy",
                "terms_of_service": "https://example.com/terms",
                "format": "json",
                "schema_version": "1.1",
                "checksum": "a1b2c3d4e5f6",
                "owner": "John Doe",
                "license": "MIT",
                "validity_start": "2023-08-01",
                "validity_end": "2024-07-31",
            }
            loader_settings =  {
            "format": "PDF",
            "source": "url",
            "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
            }
            load_jack_london = await memory._add_semantic_memory(observation = query, loader_settings=loader_settings, params=params)
            memory_loaded = await memory._fetch_semantic_memory(observation=query, params=None)
            return memory_loaded["data"]["Get"]["SEMANTICMEMORY"][0]["text"]

        # return load_jack_london
        #
        # modulator = {"relevance": 0.0, "saliency": 0.0, "frequency": 0.0}
        # # #
        # run_main_buffer = await memory._create_buffer_context(
        #     user_input="I want to know how does Buck adapt to life in the wild and then have that info translated to german ",
        #     params=params,
        #     attention_modulators=modulator,
        # )

    async def generate_chatgpt_output(query:str, context:str=None):
        if context is None:
            context = await generate_context(query=query)
            # print(context)
        else:
            pass
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": f"{context}"},
                {"role": "user", "content": query}
            ]
        )
        llm_output = response.choices[0].message.content
        # print(llm_output)
        return llm_output

    with open('base_test_set.json', 'r') as f:
        data = json.load(f)
    #
    async def test_overall_score(query:str, output:str=None, expected_output:str=None, context:str=None, context_type:str=None):
        if context_type == "gpt_search":
            context = ""
        elif context_type == "base_memory_context":
            context = await generate_context(query=query)
            output = context
        elif context_type == "hybrid_search":
            context = await generate_context(query=query)
            output = await generate_chatgpt_output(query)
        elif context_type == "memory_search":
            pass

        metric = OverallScoreMetric()
        score = metric.measure(
            query=query,
            output=output,
            expected_output=expected_output,
            context=context
        )
        print('here is the score', score)

        return score

    # await generate_chatgpt_output(query=" When was call of the wild written?")
    scores = {}
    for key, item in data.items():
        question = item['question']
        expected_ans = item['answer']
        values = await test_overall_score(query=question, expected_output=expected_ans, context_type="hybrid_search")
        scores[key] = values

    print(scores)




if __name__ == "__main__":
    import asyncio

    asyncio.run(main())