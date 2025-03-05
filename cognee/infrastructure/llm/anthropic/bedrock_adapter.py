from typing import Type, Optional
from pydantic import BaseModel
import instructor
from anthropic import AnthropicBedrock

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt


class AnthropicBedrockAdapter(LLMInterface):
    """
    Adapter for Anthropic API via AWS Bedrock
    
    Authentication is handled through AWS credentials, which can be provided in several ways:
    1. AWS profile name
    2. AWS region
    3. AWS access key and secret key
    4. AWS session token
    
    For more information, see: https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html
    """

    name = "AnthropicBedrock"
    model: str

    def __init__(
        self, 
        max_tokens: int, 
        model: str = None, 
        aws_profile: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Initialize the AnthropicBedrock adapter.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            model: The model name to use (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            aws_profile: Optional AWS profile name
            aws_region: Optional AWS region
            aws_access_key: Optional AWS access key
            aws_secret_key: Optional AWS secret key
            aws_session_token: Optional AWS session token
        """
        # Initialize the AnthropicBedrock client with AWS credentials
        bedrock_client_kwargs = {}
        if aws_profile:
            bedrock_client_kwargs["aws_profile"] = aws_profile
        if aws_region:
            bedrock_client_kwargs["aws_region"] = aws_region
        if aws_access_key:
            bedrock_client_kwargs["aws_access_key"] = aws_access_key
        if aws_secret_key:
            bedrock_client_kwargs["aws_secret_key"] = aws_secret_key
        if aws_session_token:
            bedrock_client_kwargs["aws_session_token"] = aws_session_token
            
        self.aclient = instructor.patch(
            create=AnthropicBedrock(**bedrock_client_kwargs).messages.create, 
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
