# Run multiple searches and print the results

result_1 = await cognee.search(
    "Who taught Potions at Hogwarts at time Albus Dumbledore was the headmaster?",
    datasets=["cognee-basics"],
)

# Print the result so you can see it in the notebook output
print(result_1)


result_2 = await cognee.search(
    "How to defeat Voldemort?",
    datasets=["cognee-basics"],
)

print(result_2)
