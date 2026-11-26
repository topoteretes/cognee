"""fastembed's text embedding models, with the one cognee adds to them.

fastembed only loads a model it has a description of, and ``pplx-embed-v1-0.6b``
(the local default, see ``DEFAULT_LOCAL_EMBEDDING_MODEL``) is not in its list. It is
registered here when this module is imported, which is once per process. Everything
that reads fastembed's list reads it through ``fastembed_text_models``, so the model
is always in it.
"""

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

PPLX_EMBED_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"

# fastembed refuses a name it already has: this module loaded twice, the model
# registered by the application, or a fastembed release that lists it itself.
if not any(model["model"] == PPLX_EMBED_MODEL for model in TextEmbedding.list_supported_models()):
    TextEmbedding.add_custom_model(
        model=PPLX_EMBED_MODEL,
        sources=ModelSource(hf=PPLX_EMBED_MODEL),
        dim=1024,
        # The 8-bit weights: 0.7 GB instead of the 2.4 GB of the full-precision file.
        model_file="onnx/model_quantized.onnx",
        additional_files=["onnx/model_quantized.onnx_data"],
        size_in_gb=0.71,
        # The graph mean-pools the token vectors itself: ``pooler_output`` is the text's
        # vector, so fastembed must not pool again. It is not unit length; fastembed
        # normalizes it, as it does for every model in its list.
        pooling=PoolingType.DISABLED,
        output_name="pooler_output",
        normalization=True,
        license="mit",
        description="Text embeddings, multilingual, 32000 input tokens, no prefix needed, 2026",
    )


def fastembed_text_models() -> list[dict]:
    """The description of every text embedding model fastembed can load."""
    return TextEmbedding.list_supported_models()
