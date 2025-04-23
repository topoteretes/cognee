from modal import App

app = App("cognee_distributed_test")


@app.function()
def sum_distributed(numbers: list):
    result = sum(numbers)

    return result


@app.local_entrypoint()
def main():
    sum = 0
    numbers = range(100)
    batch_size = 10

    local_sum = sum_distributed.local(numbers=numbers)

    print(f"Local sum: {local_sum}")

    batches = [list(numbers[i : i + batch_size]) for i in range(0, len(numbers), batch_size)]

    for result in sum_distributed.map(batches):
        sum += result

    print(f"Distributed sum: {sum}")
