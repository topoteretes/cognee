# Add Guido's contributions to data
await cognee.add(
    "d18g11dwdlgfey.cloudfront.net/tutorials/python-development-with-cognee/data/guido_contributions.json",
    node_set=["guido_data"],
    dataset_name="python-development-with-cognee",
)

# Cognify added data into a knowledge graph
await cognee.cognify(datasets=["python-development-with-cognee"], temporal_cognify=True)

# Search the knowledge graph
results = await cognee.search("Show me commits", datasets=["python-development-with-cognee"])
print(results)
