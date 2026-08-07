import asyncio
import os
import sys

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
# 3. Pass a video path as the first argument.
#
# The video's audio track is transcribed with per-segment [HH:MM:SS] timestamps
# inlined into the text, so they survive chunking and stay searchable.
#
# Supported formats: mp4, m4v, mov, webm, mkv, avi. ffmpeg is optional:
#   - `.mp4` and `.webm` are transcribed directly, no ffmpeg needed.
#   - Other containers (`.mov`, `.mkv`, `.avi`, `.m4v`) need ffmpeg to extract
#     the audio track. Install a system ffmpeg and make sure it is on your PATH.


async def main():
    if len(sys.argv) < 2:
        print("Usage: python video_processing_example.py /path/to/video.mp4")
        return

    video_file_path = sys.argv[1]

    if not os.path.exists(video_file_path):
        print(f"No video found at '{video_file_path}'. Pass a valid video path and rerun.")
        return

    # Create a clean slate for cognee -- reset data and system state
    await cognee.forget(everything=True)

    # cognee transcribes the video's audio track (with inline [HH:MM:SS]
    # timestamps) and builds a knowledge graph from the transcript.
    await cognee.remember([video_file_path], self_improvement=False)

    # Query cognee for a summary of what the video is about
    search_results = await cognee.recall(
        query_type=SearchType.SUMMARIES,
        query_text="What is this video about?",
    )

    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
