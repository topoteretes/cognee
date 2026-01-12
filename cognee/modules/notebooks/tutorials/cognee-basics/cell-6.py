# And after the cognification, search the knowledge graph

result = await cognee.search(
    "Which characters belong to Gryffindor?",
    datasets=["cognee-basics"],
)

# Print the result so you can see it in the notebook output
print(result)
