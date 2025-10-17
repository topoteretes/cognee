"""
Run writer and reader in separate subprocesses to test Kuzu locks.
"""

import subprocess
import time
import os


def main():
    print("=== Kuzu Subprocess Lock Test ===")
    print("Starting writer and reader in separate subprocesses...")
    print("Writer will hold the database lock, reader should block or fail\n")

    start_time = time.time()

    # Start writer subprocess
    writer_process = subprocess.Popen([os.sys.executable, "writer.py"])

    reader_process = subprocess.Popen([os.sys.executable, "reader.py"])

    # Wait for both processes to complete
    writer_process.wait()
    reader_process.wait()

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
