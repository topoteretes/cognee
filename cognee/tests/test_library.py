import os
import logging
import pathlib
import cognee
from cognee.api.v1.search import SearchType

logging.basicConfig(level = logging.DEBUG)

async def  main():
    data_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_library")).resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_library")).resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata = True)

    dataset_name = "artificial_intelligence"

    ai_text_file_path = os.path.join(pathlib.Path(__file__).parent, "test_data/artificial-intelligence.pdf")
    await cognee.add([ai_text_file_path], dataset_name)

    text = """A large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process. LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
    LLMs are artificial neural networks. The largest and most capable, as of March 2024, are built with a decoder-only transformer-based architecture while some recent implementations are based on other architectures, such as recurrent neural network variants and Mamba (a state space model).
    Up to 2020, fine tuning was the only way a model could be adapted to be able to accomplish specific tasks. Larger sized models, such as GPT-3, however, can be prompt-engineered to achieve similar results.[6] They are thought to acquire knowledge about syntax, semantics and "ontology" inherent in human language corpora, but also inaccuracies and biases present in the corpora.
    Some notable LLMs are OpenAI's GPT series of models (e.g., GPT-3.5 and GPT-4, used in ChatGPT and Microsoft Copilot), Google's PaLM and Gemini (the latter of which is currently used in the chatbot of the same name), xAI's Grok, Meta's LLaMA family of open-source models, Anthropic's Claude models, Mistral AI's open source models, and Databricks' open source DBRX.
    """

    await cognee.add([text], dataset_name)

    await cognee.cognify([dataset_name])

    from cognee.infrastructure.databases.vector import get_vector_engine
    vector_engine = get_vector_engine()
    random_node = (await vector_engine.search("entities", "AI"))[0]
    random_node_name = random_node.payload["name"]

    search_results = await cognee.search(SearchType.INSIGHTS, query = random_node_name)
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(SearchType.CHUNKS, query = random_node_name)
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(SearchType.SUMMARIES, query = random_node_name)
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\n\Extracted summaries are:\n")
    for result in search_results:
        print(f"{result}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main(), debug=True)
