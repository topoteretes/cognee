import subprocess
import json
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params_file", type=str, required=True, help="Which dataset to evaluate on"
    )
    parser.add_argument("--out_dir", type=str, help="Dir to save eval results")

    args = parser.parse_args()

    with open(args.params_file, "r") as file:
        parameters = json.load(file)

    for metric in parameters["metric_names"]:
        params = parameters
        params["metric_names"] = [metric]

        temp_paramfile = args.params_file.replace(".json", f"_{metric}.json")
        with open(temp_paramfile, "w") as file:
            json.dump(params, file)

        command = [
            "python",
            "evals/run_qa_eval.py",
            "--params_file",
            temp_paramfile,
            "--out_dir",
            args.out_dir,
        ]

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        for line in process.stdout:
            print(line, end="")  # Print output line-by-line

        process.wait()


if __name__ == "__main__":
    main()
