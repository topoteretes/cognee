result = await cognee.search(
    query_text="What can we learn from Guido's contributions in 2025?",
    query_type=cognee.SearchType.TEMPORAL,
    datasets=["python-development-with-cognee"],
)

print(result)
