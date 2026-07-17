"""Print the text an image becomes: the vision-model transcription plus appended OCR text.

Requires LLM_API_KEY (copy `.env.template` -> `.env`) and the OCR engine:
pip install "cognee[rapidocr]".
"""

import asyncio
import os
import pathlib

from cognee.infrastructure.loaders.core.image_loader import ImageLoader

os.environ.setdefault("IMAGE_EXTRACTION_ENABLED", "true")
os.environ.setdefault("IMAGE_OCR_ENABLED", "true")


async def main():
    image_path = os.path.join(pathlib.Path(__file__).parent, "data/revenue_chart.png")
    text = await ImageLoader().load(image_path, persist=False)

    print("=== Text extracted from the image ===")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
