import asyncio
import os
import cognee
from cognee.api.v1.search import SearchType
from cognee.api.v1.cognify.cognify import get_default_tasks_with_translation
from cognee.modules.pipelines.operations.pipeline import run_pipeline

# Prerequisites:
# 1. Set up your environment with API keys for your chosen translation provider.
#    - For OpenAI: OPENAI_API_KEY
#    - For Azure: AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_ENDPOINT, AZURE_TRANSLATOR_REGION
# 2. Specify the translation provider via an environment variable (optional, defaults to "noop"):
#    COGNEE_TRANSLATION_PROVIDER="openai"  # Or "google", "azure", "langdetect"
# 3. Install any required libraries for your provider:
#    - pip install langdetect googletrans==4.0.0rc1 azure-ai-translation-text

async def main():
<<<<<<< HEAD
    """Demonstrates the translation pipeline in cognee."""
=======
    """
    Demonstrates an end-to-end translation-enabled Cognify workflow using the Cognee SDK.
    
    Performs three main steps:
    1. Resets the demo workspace by pruning stored data and system metadata.
    2. Seeds three multilingual documents, builds translation-enabled Cognify tasks using the
       provider specified by the COGNEE_TRANSLATION_PROVIDER environment variable (defaults to "noop"),
       and executes the pipeline to translate and process the documents.
       - If the selected provider is missing or invalid, the function prints the error and returns early.
    3. Issues an English search query (using SearchType.INSIGHTS) against the processed index and
       prints any returned result texts.
    
    Side effects:
    - Mutates persistent Cognee state (prune, add, cognify pipeline execution).
    - Prints status and result messages to stdout.
    
    Notes:
    - No return value.
    - Exceptions ValueError and ImportError are caught and handled by printing an error and exiting the function.
    """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
    # 1. Set up cognee and add multilingual content
    print("Setting up demo environment...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    multilingual_texts = [
        "El procesamiento de lenguaje natural (PLN) es un subcampo de la IA.",
        "Le traitement automatique du langage naturel (TALN) est un sous-domaine de l'IA.",
        "Natural language processing (NLP) is a subfield of AI.",
    ]

    print("Adding multilingual texts...")
    for text in multilingual_texts:
        await cognee.add(text)
    print("Texts added successfully.\n")

    # 2. Run the cognify pipeline with translation enabled
    provider = os.getenv('COGNEE_TRANSLATION_PROVIDER', 'noop').lower()
    print(f"Running cognify with translation provider: {provider}")
    
    try:
        # Build translation-enabled tasks and execute the pipeline
        translation_enabled_tasks = get_default_tasks_with_translation(
            translation_provider=provider
        )
        async for _ in run_pipeline(tasks=translation_enabled_tasks):
            pass
        print("Cognify pipeline with translation completed successfully.")
    except (ValueError, ImportError) as e:
        print(f"Error during cognify: {e}")
        print("Please ensure the selected provider is installed and configured correctly.")
        return

    # 3. Search for content in English
    query_text = "Tell me about NLP"
    print(f"\nSearching for: '{query_text}'")
    
    # The search should now return results from all documents, as they have been translated.
    search_results = await cognee.search(query_text, query_type=SearchType.INSIGHTS)
    
    print("\nSearch Results:")
    if search_results:
        for result in search_results:
            print(f"- {result.text}")
    else:
        print("No results found.")

if __name__ == "__main__":
    asyncio.run(main())
