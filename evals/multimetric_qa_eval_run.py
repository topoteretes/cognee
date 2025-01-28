import subprocess
import json
import argparse
import os
from typing import List
import sys


def run_command(command: List[str]):
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            if stdout_line == "" and stderr_line == "" and process.poll() is not None:
                break

            if stdout_line:
                print(stdout_line.rstrip())
            if stderr_line:
                print(f"Error: {stderr_line.rstrip()}", file=sys.stderr)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        process.stdout.close()
        process.stderr.close()


def run_evals_for_paramsfile(params_file, out_dir):
    with open(params_file, "r") as file:
        parameters = json.load(file)

    for metric in parameters["metric_names"]:
        params = parameters
        params["metric_names"] = [metric]

        temp_paramfile = params_file.replace(".json", f"_{metric}.json")
        with open(temp_paramfile, "w") as file:
            json.dump(params, file)

        command = [
            "python",
            "evals/run_qa_eval.py",
            "--params_file",
            temp_paramfile,
            "--out_dir",
            out_dir,
        ]

        run_command(command)

        if os.path.exists(temp_paramfile):
            os.remove(temp_paramfile)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params_file", type=str, required=True, help="Which dataset to evaluate on"
    )
    parser.add_argument("--out_dir", type=str, help="Dir to save eval results")

    args = parser.parse_args()

    run_evals_for_paramsfile(args.params_file, args.out_dir)


if __name__ == "__main__":
    main()
