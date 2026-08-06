import json
from os.path import basename
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class JsonListChunker(Chunker):
    """Chunk a JSON list or nested JSON document into one stringified item per chunk."""

    def __init__(self, document, get_text: callable, max_chunk_size: int, json_path: str = None):
        super().__init__(document, get_text, max_chunk_size)
        self.json_path = json_path

    def _extract_chunks(self, data, current_path="", parent_context=None):
        if parent_context is None:
            parent_context = {}
            
        if isinstance(data, list):
            for index, item in enumerate(data):
                item_path = f"{current_path}[{index}]" if current_path else f"[{index}]"
                yield from self._extract_chunks(item, item_path, parent_context)
                
        elif isinstance(data, dict):
            arrays = {k: v for k, v in data.items() if isinstance(v, list)}
            dicts = {k: v for k, v in data.items() if isinstance(v, dict)}
            scalars = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
            
            if not arrays and not dicts:
                # Leaf dict
                path_str = current_path if current_path else "$"
                yield data, path_str, parent_context
            else:
                # We have nested structures. The scalars of THIS dict form context for children.
                current_context = {**parent_context, **scalars} if scalars else parent_context
                has_yielded = False
                
                # Recurse dicts
                for k, v in dicts.items():
                    child_path = f"{current_path}.{k}" if current_path else k
                    for chunk in self._extract_chunks(v, child_path, current_context):
                        has_yielded = True
                        yield chunk
                        
                # Recurse arrays
                for k, v in arrays.items():
                    child_path = f"{current_path}.{k}" if current_path else k
                    for chunk in self._extract_chunks(v, child_path, current_context):
                        has_yielded = True
                        yield chunk
                        
                if not has_yielded:
                    path_str = current_path if current_path else "$"
                    if scalars:
                        yield scalars, path_str, parent_context
                    else:
                        yield data, path_str, parent_context
        else:
            # scalar
            path_str = current_path if current_path else "$"
            yield data, path_str, parent_context

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        parsed_json = json.loads(content)

        max_observed_chunk_size = 0
        chunk_index = 0

        for item_data, json_path, parent_context in self._extract_chunks(parsed_json, self.json_path or ""):
            if isinstance(item_data, dict):
                payload = {**parent_context, **item_data}
            elif parent_context:
                payload = {**parent_context, "value": item_data}
            else:
                payload = item_data
                
            text = json.dumps(payload)
            chunk_size = len(text.split())
            max_observed_chunk_size = max(max_observed_chunk_size, chunk_size)

            if chunk_size > self.max_chunk_size:
                logger.warning(
                    "JsonListChunker item exceeds max_chunk_size",
                    chunk_index=chunk_index,
                    chunk_size=chunk_size,
                    max_chunk_size=self.max_chunk_size,
                    document_name=document_name,
                )

            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{document_id}-{chunk_index}"),
                text=text,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=chunk_index,
                cut_type="json_list_item",
                contains=[],
                importance_weight=self.document.importance_weight,
                document_id=document_id,
                document_name=document_name,
                metadata={
                    "index_fields": ["text"],
                    "json_list_index": chunk_index,
                    "json_path": json_path,
                },
            )
            chunk_index += 1

        if max_observed_chunk_size > self.max_chunk_size:
            logger.warning(
                "JsonListChunker max item size exceeds max_chunk_size",
                max_observed_chunk_size=max_observed_chunk_size,
                max_chunk_size=self.max_chunk_size,
                document_name=document_name,
            )
