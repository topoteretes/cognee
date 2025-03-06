from typing import Type, Optional
from pydantic import BaseModel
import instructor
from anthropic import AnthropicVertex

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt


class AnthropicVertexAdapter(LLMInterface):
    """
    Adapter for Anthropic API via Google Vertex AI
    
    Authentication is handled through Google Cloud's Application Default Credentials (ADC).
    ADC looks for credentials in the following order:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to a service account key file
    2. Default service account when running in Google Cloud environments
    3. User credentials from gcloud CLI (after running 'gcloud auth application-default login')
    
    For more information, see: https://cloud.google.com/docs/authentication/application-default-credentials
    """

    name = "AnthropicVertex"
    model: str

    def __init__(self, max_tokens: int, model: str = None, project_id: Optional[str] = None, location: Optional[str] = None):
        """
        Initialize the AnthropicVertex adapter.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            model: The model name to use (e.g., "claude-3-5-sonnet-v2@20241022")
            project_id: Optional Google Cloud project ID. If not provided, will use the default from ADC.
            location: Optional Google Cloud region. If not provided, defaults to "us-central1".
        """
        # Initialize the AnthropicVertex client with optional project_id and location
        vertex_client_kwargs = {}
        if project_id:
            vertex_client_kwargs["project_id"] = project_id
        if location:
            vertex_client_kwargs["location"] = location
            
        self.aclient = instructor.patch(
            create=AnthropicVertex(**vertex_client_kwargs).messages.create, 
            mode=instructor.Mode.ANTHROPIC_TOOLS
        )
        self.model = model
        self.max_tokens = max_tokens

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        return await self.aclient(
            model=self.model,
            max_tokens=4096,
            max_retries=5,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to extract information
                from the following input: {text_input}. {system_prompt}""",
                }
            ],
            response_model=response_model,
        )

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""

        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")

        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )

        return formatted_prompt
