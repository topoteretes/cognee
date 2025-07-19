async def async_gen_zip(iterable1, async_iterable2):
    it1 = iter(iterable1)
    it2 = async_iterable2.__aiter__()

    while True:
        try:
            val1 = next(it1)
            val2 = await it2.__anext__()

            yield val1, val2
        except (StopIteration, StopAsyncIteration):
            break
