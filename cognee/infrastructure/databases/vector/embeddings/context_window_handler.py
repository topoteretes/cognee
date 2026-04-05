import asyncio  
import math  
import numpy as np  
from typing import List, Callable, Awaitable  
from cognee.shared.logging_utils import get_logger  
  
logger = get_logger(__name__)  
  
async def handle_context_window_exceeded(  
    embed_func: Callable[[List[str]], Awaitable[List[List[float]]]],  
    text: List[str],  
    max_retries: int = 3  
) -> List[List[float]]:  
    """  
    Universal context window handler for embedding engines.  
    """  
    try:  
        return await embed_func(text)  
    except Exception as error:  
        # Handle different exception types from various embedding engines  
        if "context" in str(error).lower() or "token" in str(error).lower() or "window" in str(error).lower():  
            if max_retries <= 0:  
                logger.error(f"Max retries exceeded for context window handling: {error}")  
                raise error  
                  
            if len(text) > 1:  
                # Split list in half  
                mid = math.ceil(len(text) / 2)  
                left, right = text[:mid], text[mid:]  
                left_vecs, right_vecs = await asyncio.gather(  
                    handle_context_window_exceeded(embed_func, left, max_retries - 1),  
                    handle_context_window_exceeded(embed_func, right, max_retries - 1)  
                )  
                return left_vecs + right_vecs  
            else:  
                # Split single text into thirds with overlap  
                s = text[0]  
                third = len(s) // 3  
                left_part, right_part = s[: third * 2], s[third:]  
                  
                (left_vec,), (right_vec,) = await asyncio.gather(  
                    handle_context_window_exceeded(embed_func, [left_part], max_retries - 1),  
                    handle_context_window_exceeded(embed_func, [right_part], max_retries - 1)  
                )  
                  
                # Pool embeddings  
                pooled = (np.array(left_vec) + np.array(right_vec)) / 2  
                return [pooled.tolist()]  
        else:  
            # Re-raise non-context window errors  
            raise error
