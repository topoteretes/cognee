
class LLMConfig():
    llm_provider: str = None
    llm_model: str = None
    llm_endpoint: str = None
    llm_api_key: str = None

    def to_dict(self) -> dict:
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "endpoint": self.llm_endpoint,
            "apiKey": self.llm_api_key,
        }

llm_config = LLMConfig()
