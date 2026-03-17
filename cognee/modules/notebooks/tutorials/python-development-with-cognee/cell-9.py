await cognee.add(
    "d18g11dwdlgfey.cloudfront.net/tutorials/python-development-with-cognee/data/copilot_conversations.json",
    node_set=["developer_data"],
    dataset_name="python-development-with-cognee",
)

await cognee.add(
    "d18g11dwdlgfey.cloudfront.net/tutorials/python-development-with-cognee/data/my_developer_rules.md",
    node_set=["developer_data"],
    dataset_name="python-development-with-cognee",
)

await cognee.add(
    "d18g11dwdlgfey.cloudfront.net/tutorials/python-development-with-cognee/data/zen_principles.md",
    node_set=["principles_data"],
    dataset_name="python-development-with-cognee",
)

await cognee.add(
    "d18g11dwdlgfey.cloudfront.net/tutorials/python-development-with-cognee/data/pep_style_guide.md",
    node_set=["principles_data"],
    dataset_name="python-development-with-cognee",
)

await cognee.cognify(datasets=["python-development-with-cognee"], temporal_cognify=True)

results = await cognee.search(
    "What Python type hinting challenges did I face, and how does Guido approach similar problems in mypy?",
    datasets=["python-development-with-cognee"],
)
print(results)
