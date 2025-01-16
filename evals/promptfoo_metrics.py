from evals.promptfoo_wrapper import PromptfooWrapper
import os
import yaml
import json
import shutil
from cognee.infrastructure.llm.prompts.llm_judge_prompts import llm_judge_prompts


def is_valid_promptfoo_metric(metric_name: str):
    try:
        prefix, suffix = metric_name.split(".")
    except ValueError:
        return False
    if prefix != "promptfoo":
        return False
    if suffix not in llm_judge_prompts:
        return False
    return True


class PromptfooMetric:
    def __init__(self, metric_name_list):
        promptfoo_path = shutil.which("promptfoo")
        self.wrapper = PromptfooWrapper(promptfoo_path=promptfoo_path)
        self.prompts = {}
        for metric_name in metric_name_list:
            if is_valid_promptfoo_metric(metric_name):
                self.prompts[metric_name] = llm_judge_prompts[metric_name.split(".")[1]]
            else:
                raise Exception(f"{metric_name} is not a valid promptfoo metric")

    async def measure(self, instances, context_provider):
        with open(os.path.join(os.getcwd(), "evals/promptfoo_config_template.yaml"), "r") as file:
            config = yaml.safe_load(file)

        config["defaultTest"] = {
            "assert": [
                {"type": "llm-rubric", "value": prompt, "name": metric_name}
                for metric_name, prompt in self.prompts.items()
            ]
        }

        # Fill config file with test cases
        tests = []
        for instance in instances:
            context = await context_provider(instance)
            test = {
                "vars": {
                    "name": instance["question"][:15],
                    "question": instance["question"],
                    "context": context,
                }
            }
            tests.append(test)
        config["tests"] = tests

        # Write the updated YAML back, preserving formatting and structure
        updated_yaml_file_path = os.path.join(os.getcwd(), "config_with_context.yaml")
        with open(updated_yaml_file_path, "w") as file:
            yaml.dump(config, file)

        self.wrapper.run_eval(
            prompt_file=os.path.join(os.getcwd(), "evals/promptfooprompt.json"),
            config_file=os.path.join(os.getcwd(), "config_with_context.yaml"),
            out_format="json",
        )

        file_path = os.path.join(os.getcwd(), "benchmark_results.json")

        # Read and parse the JSON file
        with open(file_path, "r") as file:
            results = json.load(file)

        scores = {}

        for result in results["results"]["results"][0]["gradingResult"]["componentResults"]:
            scores[result["assertion"]["name"]] = result["score"]

        return scores
