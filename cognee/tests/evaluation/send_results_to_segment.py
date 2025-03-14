from posthog import Posthog
import os
import uuid
import logging
import json
from dotenv import load_dotenv
import argparse
from cognee.shared.utils import setup_logging
import analytics
import datetime

load_dotenv()

setup_logging(logging.INFO)

SEGMENT_WRITE_KEY = os.getenv("SEGMENT_WRITE_KEY_EVAL")
analytics.write_key = SEGMENT_WRITE_KEY


def send_event_to_segment(results):
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

    properties = {
        f"mean_{key}": results["aggregate_metrics"][key]["mean"]
        for key in results["aggregate_metrics"].keys()
    }
    properties["created_at"] = created_at

    # Send event to Segment
    analytics.track(
        user_id="evalresults_ingest_bot",  # Unique identifier for the event
        event="cognee_eval_results",
        properties=properties,
    )

    # Ensure all events are sent
    analytics.flush()


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
    send_event_to_segment(results)


if __name__ == "__main__":
    main()
