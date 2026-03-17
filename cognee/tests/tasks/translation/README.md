# Translation Task Tests

Unit and integration tests for the multilingual content translation feature.

## Test Files

- **config_test.py** - Tests for translation configuration
  - Default configuration
  - Provider type validation
  - Confidence threshold bounds
  - Multiple provider API keys

- **detect_language_test.py** - Tests for language detection functionality
  - English, Spanish, French, German, Chinese detection
  - Confidence thresholds
  - Edge cases (empty text, short text, mixed languages)

- **providers_test.py** - Tests for translation provider implementations
  - LLM provider basic translation
  - Auto-detection of source language
  - Batch translation
  - Special characters and formatting preservation
  - Error handling

- **translate_content_test.py** - Tests for the main translate_content task
  - Basic translation workflow
  - Original text preservation
  - Multiple chunks processing
  - Language metadata creation
  - Skip translation for target language
  - Confidence threshold customization

## Running Tests

### Run all translation tests
```bash
uv run pytest cognee/tests/tasks/translation/ -v
```

### Run specific test file
```bash
uv run pytest cognee/tests/tasks/translation/detect_language_test.py -v
```

### Run tests directly (without pytest)
```bash
uv run python cognee/tests/tasks/translation/config_test.py
uv run python cognee/tests/tasks/translation/detect_language_test.py
uv run python cognee/tests/tasks/translation/providers_test.py
uv run python cognee/tests/tasks/translation/translate_content_test.py
uv run python cognee/tests/tasks/translation/integration_test.py
```

### Run all tests at once
```bash
for f in cognee/tests/tasks/translation/*_test.py; do uv run python "$f"; done
```

### Run with coverage
```bash
uv run pytest cognee/tests/tasks/translation/ --cov=cognee.tasks.translation --cov-report=html
```

## Prerequisites

- LLM API key set in environment: `LLM_API_KEY=your_key`
- Tests will be skipped if no API key is available

**Note:** The translation feature uses the same LLM model configured for other cognee tasks (via `LLM_MODEL` and `LLM_PROVIDER` environment variables). This means any LLM provider supported by cognee (OpenAI, Azure, Anthropic, Ollama, etc.) can be used for translation.

## Usage Example

```python
import cognee
from cognee.tasks.translation import translate_text

# Configure translation (optional - defaults to LLM provider)
cognee.config.set_translation_config(
    provider="llm",           # Uses configured LLM (default)
    target_language="en",     # Target language code
    confidence_threshold=0.7  # Minimum confidence for language detection
)

# Translate text directly
result = await translate_text(
    text="Bonjour le monde",
    target_language="en"
)
print(result.translated_text)  # "Hello world"
```

### Alternative Translation Providers

```python
# Use Google Cloud Translate (requires GOOGLE_TRANSLATE_API_KEY)
cognee.config.set_translation_provider("google")

# Use Azure Translator (requires AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION)
cognee.config.set_translation_provider("azure")
```

## Test Summary

| Test File | Tests | Description |
|-----------|-------|-------------|
| config_test.py | 4 | Configuration validation |
| detect_language_test.py | 10 | Language detection |
| providers_test.py | 9 | Translation providers |
| translate_content_test.py | 9 | Content translation task |
| integration_test.py | 2 | Standalone translation tests |
| **Total** | **34** | |

## Test Categories

### Configuration (4 tests)
- ✅ Default configuration values
- ✅ Provider type literal validation
- ✅ Confidence threshold bounds
- ✅ Multiple provider API keys

### Language Detection (10 tests)
- ✅ Multiple language detection (EN, ES, FR, DE, ZH)
- ✅ Confidence scoring
- ✅ Target language matching
- ✅ Short and empty text handling
- ✅ Mixed language detection

### Translation Providers (9 tests)
- ✅ Provider factory function
- ✅ LLM translation
- ✅ Batch operations
- ✅ Auto source language detection
- ✅ Long text handling
- ✅ Special characters preservation
- ✅ Error handling

### Content Translation (9 tests)
- ✅ DocumentChunk processing
- ✅ Metadata creation (LanguageMetadata, TranslatedContent)
- ✅ Original text preservation
- ✅ Multiple chunk handling
- ✅ Empty text/list handling
- ✅ Confidence threshold customization

### Integration (2 tests)
- ✅ Direct translate_text function
- ✅ Language detection functionality
