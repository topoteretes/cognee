# Multimedia processing

cognee turns multimedia files into text and builds a knowledge graph from that
text, so you can search across audio, images, and video the same way you search
documents.

| Modality | How the text is produced | Example |
| --- | --- | --- |
| Audio | Audio-track transcription (`create_transcript`) | [`guides/multimedia_audio_image_processing_example.py`](../../guides/multimedia_audio_image_processing_example.py) |
| Image | Vision caption (`transcribe_image`) | [`guides/multimedia_audio_image_processing_example.py`](../../guides/multimedia_audio_image_processing_example.py) |
| Video | Audio-track transcription with inline timestamps | `video_processing_example.py` (this folder) |

Audio and image ingestion moved to `examples/guides/` — it is a single-feature guide, and its
assets live beside it in `guides/multimedia_audio_image_processing_example_data/`. This folder now
covers video only.

## Video ingestion

A video's audio track is transcribed and written out with per-segment
`[HH:MM:SS]` timestamps inlined into the text, for example:

```
[00:00:00] Welcome to the walkthrough.
[00:00:12] First we configure the environment.
```

The timestamps are part of the text, so they survive chunking and stay
searchable. From there the transcript flows through the normal pipeline
(chunking, entity and relationship extraction, graph + vector storage), so a
video becomes queryable memory with no special handling downstream.

### Supported formats

`mp4`, `m4v`, `mov`, `webm`, `mkv`, `avi`.

### ffmpeg is optional

- `mp4` and `webm` are transcribed directly, no ffmpeg required.
- Other containers (`mov`, `mkv`, `avi`, `m4v`) need ffmpeg to extract the audio
  track first. When ffmpeg is available it is also used for `mp4`/`webm`, which
  keeps the upload small.
- Install ffmpeg and make sure it is on your `PATH`. If a container needs ffmpeg
  and none is found, cognee raises a clear error explaining how to enable it.

### Run the example

```bash
# uses the bundled sample-video.mp4 next to the script
python video_processing_example.py

# or pass your own file (.mp4 or .webm works without ffmpeg)
python video_processing_example.py /path/to/your/video.mp4
```

Requires an `LLM_API_KEY` in your `.env` (see the repository README for setup).
