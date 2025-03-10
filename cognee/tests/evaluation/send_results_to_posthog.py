from posthog import Posthog
import os
import uuid
import logging
import json
from dotenv import load_dotenv
import argparse
from cognee.shared.utils import setup_logging

load_dotenv()

setup_logging(logging.INFO)


def initialize_posthog_client():
    posthog = Posthog(
        api_key=os.getenv("POSTHOG_API_KEY_DEV"),
        host="https://eu.i.posthog.com",
    )
    posthog.debug = True
    logging.info("PostHog client initialized.")
    return posthog


def send_event_to_posthog(posthog, results):
    properties = {
        f"mean_{key}": results["aggregate_metrics"][key]["mean"]
        for key in results["aggregate_metrics"].keys()
    }
    logging.info(properties)
    posthog.capture(
        distinct_id=str(uuid.uuid4()),
        event="cognee_eval_results",
        properties=properties,
    )

    logging.info("Event sent to PostHog successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        default="metrics_output.json",
        help="The filename of the results to send to PostHog.",
    )
    args = parser.parse_args()
    with open(args.filename, "r") as f:
        results = json.load(f)
    logging.info(
        f"results loaded, mean correctness {results['aggregate_metrics']['correctness']['mean']}"
    )
    posthog = initialize_posthog_client()
    send_event_to_posthog(posthog, results)


if __name__ == "__main__":
    main()
