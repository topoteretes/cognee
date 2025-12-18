result_1 = await cognee.search(
    "Which characters belong to Gryffindor?",
    datasets=["cognee-basics"],
)

print(result_1)


result_2 = await cognee.search(
    "Who taught Potions at Hogwarts at time Albus Dumbledore was the headmaster?",
    datasets=["cognee-basics"],
)

print(result_2)


result_3 = await cognee.search(
    "How to defeat Voldemort?",
    datasets=["cognee-basics"],
)

print(result_3)
