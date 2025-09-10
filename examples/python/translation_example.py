import asyncio
import os
import cognee
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.api.v1.search import SearchType
from cognee.api.v1.cognify.cognify import get_default_tasks_with_translation
from cognee.modules.pipelines.operations.pipeline import run_pipeline
from typing import Tuple

# Shared multilingual sample texts
MULTILINGUAL_TEXTS = [
    """
    El procesamiento de lenguaje natural (PLN) es un subcampo interdisciplinario 
    de las ciencias de la computación y la recuperación de información.
    """,
    """
    Le traitement automatique du langage naturel (TALN) est un domaine 
    interdisciplinaire de l'informatique et de la recherche d'information.
    """,
    """
    Natural language processing (NLP) is an interdisciplinary subfield 
    of computer science and information retrieval.
    """
]

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your API keys to the `.env` file:
#    OPENAI_API_KEY = "your_openai_key_here"            # For OpenAI provider
#    AZURE_TRANSLATOR_KEY = "your_azure_key_here"       # For Azure provider  
#    AZURE_TRANSLATOR_ENDPOINT = "your_azure_endpoint"  # Optional for Azure
#    AZURE_TRANSLATOR_REGION = "your_azure_region"      # Optional for Azure
# 3. Optionally set translation provider via environment variable:
#    COGNEE_TRANSLATION_PROVIDER = "openai" | "google" | "azure" | "langdetect" | "noop"  # tip: "noop" is the safest default
# 4. Optionally install translation libraries:
#    pip install langdetect googletrans==4.0.0rc1 azure-ai-translation-text


async def setup_demo_data():
    """Set up clean demo environment with multilingual data."""
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    print("Adding multilingual texts to cognee:")
    for i, text in enumerate(MULTILINGUAL_TEXTS, 1):
        print(f"\nText {i}: {text.strip()}")
        # Add each text as a separate document
        await cognee.add(text, dataset_name=f"multilingual_demo_{i}")
    print("\nAll texts added successfully.\n")


async def demo_standard_pipeline():
    """Demonstrate standard pipeline without translation."""
    print("=" * 60)
    print("DEMO 1: Standard Pipeline (No Translation)")
    print("=" * 60)
    
    print("Running cognify with standard pipeline...\n")
    print("Pipeline steps:")
    print("1. Classifying documents")
    print("2. Checking permissions")  
    print("3. Extracting text chunks")
    print("4. Generating knowledge graph")
    print("5. Summarizing text")
    print("6. Adding data points\n")

    # Use standard cognify (no translation)
    await cognee.cognify()
    print("Standard pipeline complete.\n")

    # Search for content
    query_text = "Tell me about NLP"
    print(f"Searching for: '{query_text}'")
    search_results = await cognee.search(query_text, query_type=SearchType.INSIGHTS)
    
    print("Search results from standard pipeline:")
    for result in search_results[:2]:  # Show first 2 results
        print(f"- {result}")
    print()

    # Reset for second demo
    print("Resetting data for translation demo...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # Re-add the texts
    for i, text in enumerate(MULTILINGUAL_TEXTS, 1):
        await cognee.add(text, dataset_name=f"multilingual_demo_translation_{i}")

    # Demonstration 2: Using translation pipeline
    print("\n" + "=" * 60)
    print("DEMO 2: Pipeline with Translation")
    print("=" * 60)
    
    print("Running cognify with translation pipeline...\n")
    print("Enhanced pipeline steps:")
    print("1. Classifying documents")
    print("2. Checking permissions")
    print("3. Extracting text chunks") 
    print("4. 🌍 TRANSLATING non-English content to English")
    print("   - Detects language automatically")
    print("   - Translates if confidence > 80%")
    print("   - Stores original and translated text")
    print("   - Uses translated text for graph extraction")
    print("5. Generating knowledge graph (from translated content)")
    print("6. Summarizing text")
    print("7. Adding data points\n")

    # Get translation-enabled tasks with configurable provider
    provider = os.getenv("COGNEE_TRANSLATION_PROVIDER", "openai")
    try:
        tasks_with_translation = get_default_tasks_with_translation(
            translation_provider=provider
        )
    except ValueError as e:
        print(f"{e}\nFalling back to 'noop' provider for the demo.")
        tasks_with_translation = get_default_tasks_with_translation(
            translation_provider="noop"
        )
    
    # Run pipeline with translation
    print("Processing multilingual content...")
    async for result in run_pipeline(
        tasks=tasks_with_translation,
        # Use the dataset names added above
        datasets=[
            "multilingual_demo_translation_1",
            "multilingual_demo_translation_2",
            "multilingual_demo_translation_3",
        ],
    ):
        if hasattr(result, 'payload'):
            print(f"Processing: {type(result).__name__}")
    
    print("Translation pipeline complete.\n")

    # Search the translated content
    print(f"Searching translated content for: '{query_text}'")
    search_results_translated = await cognee.search(query_text, query_type=SearchType.INSIGHTS)
    
    print("Search results from translation pipeline:")
    for result in search_results_translated[:3]:  # Show first 3 results
        print(f"- {result}")
    print()

    # Demonstration 3: Custom Translation Provider
    print("\n" + "=" * 60)
    print("DEMO 3: Custom Translation Provider")
    print("=" * 60)
    
    # Import the translation system
    from cognee.tasks.translation import register_translation_provider, TranslationProvider
    
    class MockTranslationProvider(TranslationProvider):
        """Example custom translation provider that adds [TRANSLATED] prefix."""
        
        async def detect_language(self, text: str) -> Tuple[str, float]:
            # Simple mock detection
            if "El " in text or "es " in text:
                return "es", 0.9
            if "Le " in text or "du " in text:
                return "fr", 0.9
            return "en", 0.9
                
        async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
            # Mock translation - just adds a prefix
            if target_language == "en":
                return f"[MOCK TRANSLATED] {text}", 0.8
            return text, 0.0
    
    # Register the custom provider
    register_translation_provider("mock", MockTranslationProvider)
    
    print("Registered custom 'mock' translation provider")
    
    # Show available providers
    from cognee.tasks.translation import get_available_providers
    print(f"Available providers: {get_available_providers()}")
    
    # Use custom provider
    print("\nUsing custom provider to process Spanish text...")
    spanish_text = "La inteligencia artificial es fascinante."
    
    await cognee.prune.prune_data()
    await cognee.add(spanish_text, dataset_name="custom_provider_demo")
    
    # Get tasks with custom provider and run the pipeline
    tasks_with_mock = get_default_tasks_with_translation(
        translation_provider="mock"
    )
    
    async for _ in run_pipeline(
        tasks=tasks_with_mock,
        datasets=["custom_provider_demo"],
    ):
        pass
    
    print("Custom provider demo complete!")

    print("\n" + "=" * 60)
    print("TRANSLATION DEMO SUMMARY")
    print("=" * 60)
    print("✅ Standard pipeline: Processes content as-is")
    print("✅ Translation pipeline: Auto-detects and translates non-English content")
    print("✅ Custom providers: Extensible system for specialized translation needs")
    print("✅ Metadata preservation: Original text and translation metadata stored")
    print("✅ Seamless integration: Works with existing Cognee search and analysis")
    print("\nTranslation enhances Cognee's multilingual capabilities! 🌍")


async def run_all_demos():
    """Run all translation demos."""
    await setup_demo_data()
    await demo_standard_pipeline()
    print("\nAll demos completed successfully!")


if __name__ == "__main__":
    # Set up logging to see what's happening
    setup_logging(ERROR)
    
    print("🌍 Cognee Translation Demo")
    print("This example demonstrates automatic language detection and translation")
    print("for multilingual content processing.\n")
    
    try:
        asyncio.run(run_all_demos())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except (ValueError, RuntimeError) as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure you have:")
        print("1. Set up your .env file with OpenAI API key")
        print("2. Installed required dependencies")
        print("3. Run from the cognee project root directory")
        raise
