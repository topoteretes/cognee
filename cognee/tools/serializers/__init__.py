from .openai import for_openai
from .anthropic import for_anthropic
from .generic import for_generic
from .langchain import for_langchain
from .crewai import for_crewai

__all__ = ["for_openai", "for_anthropic", "for_generic", "for_langchain", "for_crewai"]
