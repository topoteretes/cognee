"""fastembed pads every batch to its longest text (SDK-810).

fastembed keeps a fixed padding length stored in a model's tokenizer.json
(all-MiniLM-L6-v2, gte-base: 128) and pads only the texts below it, so a batch
mixing texts above and below that length is a ragged array and fails. The engine
switches such a tokenizer to fastembed's own default: pad to the batch's longest.
"""

from unittest.mock import MagicMock, patch

from cognee.infrastructure.databases.vector.embeddings import FastembedEmbeddingEngine as module
from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
    pad_to_batch_longest,
)


def _model(padding):
    embedding_model = MagicMock()
    embedding_model.model.tokenizer.padding = padding
    return embedding_model


def test_a_stored_fixed_length_becomes_batch_longest_with_the_same_pad_token():
    model = _model(
        {
            "length": 128,
            "pad_to_multiple_of": None,
            "pad_id": 0,
            "pad_token": "[PAD]",
            "pad_type_id": 0,
            "direction": "right",
        }
    )
    pad_to_batch_longest(model)
    model.model.tokenizer.enable_padding.assert_called_once_with(
        direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]"
    )


def test_batch_longest_padding_is_left_alone():
    model = _model(
        {"length": None, "pad_id": 0, "pad_token": "[PAD]", "pad_type_id": 0, "direction": "right"}
    )
    pad_to_batch_longest(model)
    model.model.tokenizer.enable_padding.assert_not_called()


def test_no_padding_or_no_tokenizer_is_left_alone():
    model = _model(None)
    pad_to_batch_longest(model)
    model.model.tokenizer.enable_padding.assert_not_called()
    pad_to_batch_longest(MagicMock(model=None))  # nothing to adjust, nothing raised


def test_the_engine_switches_a_loaded_models_fixed_padding_on_init():
    text_embedding = MagicMock()
    tokenizer = text_embedding.return_value.model.tokenizer
    tokenizer.padding = {
        "length": 128,
        "pad_id": 0,
        "pad_token": "[PAD]",
        "pad_type_id": 0,
        "direction": "right",
    }
    tokenizer.truncation = {"max_length": 256}
    tokenizer.num_special_tokens_to_add.return_value = 2
    with (
        patch.object(module, "TextEmbedding", text_embedding),
        patch.object(module, "fastembed_model_cached", return_value=(True, "/cache", "90 MB")),
        patch.object(FastembedEmbeddingEngine, "get_tokenizer", return_value=MagicMock()),
    ):
        FastembedEmbeddingEngine(model="sentence-transformers/all-MiniLM-L6-v2", dimensions=384)
    tokenizer.enable_padding.assert_called_once_with(
        direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]"
    )
