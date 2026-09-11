import logging

import modal

from modal_apps.modal_image import image

logger = logging.getLogger(__name__)

APP_NAME = "volume-reader"
VOLUME_NAME = "qa-benchmarks"

# Create volume reference
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={f"/{VOLUME_NAME}": volume},
    timeout=300,
    cpu=1,
    memory=1024,
)
def get_answers_files(benchmark_folder: str):
    """Get list of JSON files from the answers folder in a benchmark directory."""
    import os

    answers_folder = f"/{VOLUME_NAME}/{benchmark_folder}/answers"
    print(f"📁 Reading contents of answers folder: {answers_folder}")

    # Reload volume to get latest changes
    volume.reload()

    try:
        if not os.path.exists(answers_folder):
            print(f"❌ Answers folder does not exist: {answers_folder}")
            return []

        contents = os.listdir(answers_folder)
        print(f"📋 Found {len(contents)} items in answers folder:")

        # Filter for JSON files
        json_files = []
        for item in contents:
            if item.endswith(".json"):
                json_files.append(item)
                item_path = f"{answers_folder}/{item}"
                size = os.path.getsize(item_path)
                print(f"  📄 {item} (file, {size} bytes)")

        print(f"✅ Found {len(json_files)} JSON files in answers folder")
        return json_files

    except FileNotFoundError:
        print("📭 Answers folder is empty or doesn't exist")
        return []
    except Exception as e:
        logger.debug("Falling back to [] after error in get_answers_files", exc_info=True)
        print(f"❌ Error reading answers folder: {e}")
        return []


@app.function(
    volumes={f"/{VOLUME_NAME}": volume},
    timeout=300,
    cpu=1,
    memory=1024,
)
def calculate_qa_metrics(benchmark_folder: str, filename: str):
    """Calculate QA metrics for a JSON file using cognee evaluation framework."""
    import asyncio
    import json
    import os

    import cognee
    from cognee.eval_framework.eval_config import EvalConfig
    from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

    answers_folder = f"/{VOLUME_NAME}/{benchmark_folder}/answers"
    deepeval_folder = f"/{VOLUME_NAME}/{benchmark_folder}/deepeval_evaluated"
    directllm_folder = f"/{VOLUME_NAME}/{benchmark_folder}/directllm_evaluated"
    evaluated_folder = f"/{VOLUME_NAME}/{benchmark_folder}/evaluated"

    # Create evaluation folders if they don't exist
    os.makedirs(deepeval_folder, exist_ok=True)
    os.makedirs(directllm_folder, exist_ok=True)
    os.makedirs(evaluated_folder, exist_ok=True)

    input_file_path = f"{answers_folder}/{filename}"
    print(f"📄 Processing file: {filename}")

    try:
        with open(input_file_path, "r") as f:
            data = json.load(f)

        print(f"✅ Successfully loaded {filename}")
        print(f"📊 JSON structure: {type(data)}")

        # Create output filenames for metrics
        base_name = filename.replace(".json", "")
        deepeval_filename = f"evaluated_{base_name}.json"
        directllm_filename = f"evaluated_{base_name}.json"
        unified_filename = f"evaluated_{base_name}.json"
        deepeval_path = f"{deepeval_folder}/{deepeval_filename}"
        directllm_path = f"{directllm_folder}/{directllm_filename}"
        unified_path = f"{evaluated_folder}/{unified_filename}"

        print("📈 Calculating metrics, outputs will be:")
        print("  - Deepeval: {deepeval_filename}")
        print("  - DirectLLM: {directllm_filename}")
        print("  - Unified: {unified_filename}")

        # Deepeval config for evaluation
        eval_config_deepeval = EvalConfig(
            answers_path=input_file_path, metrics_path=deepeval_path, evaluating_contexts=False
        )

        # DirectLLM config for evaluation
        eval_config_direct = EvalConfig(
            answers_path=input_file_path,
            metrics_path=directllm_path,
            evaluating_contexts=False,
            evaluation_engine="DirectLLM",
            evaluation_metrics=["correctness"],
        )

        # Run both evaluations
        async def run_eval():
            print("🔄 Running Deepeval evaluation...")
            await run_evaluation(eval_config_deepeval.to_dict())
            print("✅ Deepeval evaluation completed")

            print("🔄 Running DirectLLM evaluation...")
            await run_evaluation(eval_config_direct.to_dict())
            print("✅ DirectLLM evaluation completed")

        # Execute the evaluations
        asyncio.run(run_eval())

        print(f"✅ Both evaluations completed for {filename}")

        # Verify output files were created and merge them
        if os.path.exists(deepeval_path) and os.path.exists(directllm_path):
            print("🔄 Merging evaluation results...")

            # Read both evaluation files
            with open(deepeval_path, "r") as f:
                deepeval_results = json.load(f)

            with open(directllm_path, "r") as f:
                directllm_results = json.load(f)

            # Create unified results
            unified_results = []

            for i, (deepeval_item, directllm_item) in enumerate(
                zip(deepeval_results, directllm_results)
            ):
                # Ensure both items have the same question and answer
                if (
                    deepeval_item["question"] != directllm_item["question"]
                    or deepeval_item["answer"] != directllm_item["answer"]
                    or deepeval_item["golden_answer"] != directllm_item["golden_answer"]
                ):
                    print(f"⚠️ Warning: Mismatch in item {i} between evaluation results")
                    continue

                # Create unified item with all metrics
                unified_item = {
                    "question": deepeval_item["question"],
                    "answer": deepeval_item["answer"],
                    "golden_answer": deepeval_item["golden_answer"],
                    "metrics": {
                        "directllm_correctness": directllm_item["metrics"]["correctness"]["score"],
                        "deepeval_correctness": deepeval_item["metrics"]["correctness"]["score"],
                        "EM": deepeval_item["metrics"]["EM"]["score"],
                        "f1": deepeval_item["metrics"]["f1"]["score"],
                    },
                }
                unified_results.append(unified_item)

            # Save unified results
            with open(unified_path, "w") as f:
                json.dump(unified_results, f, indent=2)

            print(f"✅ Unified results saved to: {unified_filename}")
            print(f"📊 Processed {len(unified_results)} items")

        else:
            print("❌ One or both evaluation files not found, skipping merge")
            if not os.path.exists(deepeval_path):
                print("⚠️ Deepeval output file not found after evaluation")
            if not os.path.exists(directllm_path):
                print("⚠️ DirectLLM output file not found after evaluation")

    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filename}: {e}")
    except Exception as e:
        logger.debug("Ignoring exception in calculate_qa_metrics", exc_info=True)
        print(f"❌ Error processing {filename}: {e}")


@app.local_entrypoint()
def main(benchmark_folder: str | None = None, limit: int | None = None):
    """Entry point that triggers evaluation for a specific benchmark folder."""
    print(f"🚀 Starting evaluation for benchmark folder: {benchmark_folder}")
    print(f"📏 Processing limit: {limit if limit else 'all'} files")

    # Get JSON files from answers folder
    json_files = get_answers_files.remote(benchmark_folder)

    if not json_files:
        print("❌ No JSON files found to evaluate")
        return

    # Process files up to the limit
    files_to_process = json_files[:limit] if limit else json_files
    print(f"🔄 Processing {len(files_to_process)} files...")

    # Fire-and-forget approach using spawn
    for filename in files_to_process:
        calculate_qa_metrics.spawn(benchmark_folder, filename)

    print(f"✅ {len(files_to_process)} evaluation task(s) submitted successfully.")
