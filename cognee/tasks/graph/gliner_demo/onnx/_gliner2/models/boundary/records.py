# Not copied: gliner2's record decoder is training-coupled and cognee's
# entity + relation schema never decodes records.


def decode_group(*args, **kwargs):
    raise NotImplementedError("record extraction is not supported by the torch-free GLiNER path")
