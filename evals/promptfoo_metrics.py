from evals.promptfoo_wrapper import PromptfooWrapper
import os
from deepeval.test_case import LLMTestCase
import yaml
import json


class PromptfooMetric:
    def __init__(self, judge_prompt):
        self.wrapper = PromptfooWrapper(promptfoo_path="/opt/homebrew/bin/promptfoo")
        self.judge_prompt = judge_prompt

    async def measure(self, instances, context_provider):
        with open(os.path.join(os.getcwd(), "evals/promptfoo_config_template.yaml"), "r") as file:
            config = yaml.safe_load(file)

        config["defaultTest"] = [{"assert": {"type": "llm_rubric", "value": self.judge_prompt}}]

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

        self.score = results["results"]["prompts"][0]["metrics"]["score"]

        return self.score
