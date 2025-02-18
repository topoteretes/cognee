from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityCompletionConfig(BaseSettings):
    """Configuration for entity completion pipeline."""

    entity_extractor: str = "SpacyEntityExtractor"  # Options: Implement BaseEntityExtractor
    context_getter: str = "VectorDBContextGetter"  # Options: Implement BaseContextGetter
    system_prompt_template: str = "answer_simple_question.txt"
    user_prompt_template: str = "context_for_question.txt"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")
