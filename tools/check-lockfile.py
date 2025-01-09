import sys

# File and string to search for
lockfile_name = "poetry.lock"
hash_string = "hash = "
threshold = 100

try:
    count = 0
    with open(lockfile_name, "r", encoding="utf8") as file:
        for line in file:
            if hash_string in line:
                count += 1
                if count >= threshold:
                    print(
                        f"Success: Found '{hash_string}' more than {threshold} times in {lockfile_name}."
                    )
                    sys.exit(0)

    # If the loop completes without early exit, it means the threshold was not reached
    print(
        f"Error: The string '{hash_string}' appears less than {threshold} times in {lockfile_name}, please make sure you are using an up to date poetry version."
    )
    sys.exit(1)

except FileNotFoundError:
    print(f"Error: File {lockfile_name} does not exist.")
    sys.exit(1)
