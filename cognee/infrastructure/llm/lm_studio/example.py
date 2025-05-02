"""Example usage of the LM Studio adapter in Cognee.

This script demonstrates how to use the LM Studio adapter for various tasks.
Before running, make sure:
1. LM Studio is installed and running
2. The API server is started in LM Studio
3. A model is loaded in LM Studio

Usage:
    python -m cognee.infrastructure.llm.lm_studio.example
"""

import asyncio
import os
from pydantic import BaseModel, Field
from typing import List

from cognee.infrastructure.llm.lm_studio.adapter import LMStudioAdapter


class MovieRecommendation(BaseModel):
    """Model for movie recommendations."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    genre: str = Field(description="The genre of the movie")
    reason: str = Field(description="Why this movie is recommended")


class MovieRecommendations(BaseModel):
    """Model for a list of movie recommendations."""
    recommendations: List[MovieRecommendation] = Field(
        description="List of recommended movies"
    )


async def main():
    """Run example LM Studio adapter usage."""
    # Configure the adapter
    # In a real application, these would come from environment variables
    adapter = LMStudioAdapter(
        endpoint="http://localhost:1234/v1",
        api_key="lm-studio",  # Can be any string
        model="llama-3.2-1b-instruct",  # Replace with your loaded model
        max_tokens=2000,
        streaming=False,
        temperature=0.7,
    )

    # Check connection
    if not adapter.check_connection():
        print("❌ Could not connect to LM Studio API. Make sure it's running.")
        return

    print("✅ Connected to LM Studio API")

    # Get available models
    try:
        models = adapter.get_available_models()
        print(f"📋 Available models: {[model.id for model in models]}")
    except Exception as e:
        print(f"❌ Error getting models: {e}")

    # Generate structured output
    try:
        system_prompt = """
        You are a movie recommendation system. Based on the user's preferences,
        recommend movies they might enjoy. Format your response as a JSON object.
        """
        
        user_input = """
        I enjoy science fiction movies with philosophical themes like Blade Runner
        and Arrival. I also like Christopher Nolan films.
        """
        
        print("\n🎬 Generating movie recommendations...")
        result = await adapter.acreate_structured_output(
            user_input, system_prompt, MovieRecommendations
        )
        
        print("\nRecommended Movies:")
        for movie in result.recommendations:
            print(f"- {movie.title} ({movie.year}) - {movie.genre}")
            print(f"  Reason: {movie.reason}\n")
    except Exception as e:
        print(f"❌ Error generating structured output: {e}")

    # Try image analysis if a file is available
    image_path = os.path.join(os.path.dirname(__file__), "example_image.jpg")
    if os.path.exists(image_path):
        try:
            print("\n🖼️ Analyzing image...")
            image_description = adapter.transcribe_image(
                image_path, 
                prompt="Describe what you see in this image in detail."
            )
            print(f"Image description: {image_description}")
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
    else:
        print(f"\n⚠️ Image file not found at {image_path}. Skipping image analysis.")

    # Try embeddings
    try:
        print("\n🔤 Generating embeddings...")
        embeddings = adapter.create_embeddings("This is a test of the embedding functionality.")
        print(f"Generated {len(embeddings)} embedding dimensions")
        print(f"First 5 values: {embeddings[:5]}")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")


if __name__ == "__main__":
    asyncio.run(main())
