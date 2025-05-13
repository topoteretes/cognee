import modal
import os
import json
import asyncio
import datetime
from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.metrics_dashboard import create_dashboard

logger = get_logger()
vol = modal.Volume.from_name("baseline_results", create_if_missing=True)


def read_and_combine_metrics(eval_params: dict) -> dict:
    """Read and combine metrics files into a single result dictionary."""
    try:
        with open(eval_params["metrics_path"], "r") as f:
            metrics = json.load(f)
        with open(eval_params["aggregate_metrics_path"], "r") as f:
            aggregate_metrics = json.load(f)

        return {
            "task_getter_type": eval_params["task_getter_type"],
            "number_of_samples": eval_params["number_of_samples_in_corpus"],
            "metrics": metrics,
            "aggregate_metrics": aggregate_metrics,
        }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading metrics files: {e}")
        return None


app = modal.App("modal-run-eval")

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env(
        {
            "ENV": os.getenv("ENV"),
            "LLM_API_KEY": os.getenv("LLM_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        }
    )
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2", "deepeval", "gdown", "plotly")
)


@app.function(image=image, concurrency_limit=6, timeout=86400, retries=1, volumes={"/data": vol})
async def modal_run_eval2(eval_params=None):
    """Runs evaluation pipeline and returns combined metrics results."""
    if eval_params is None:
        eval_params = EvalConfig().to_dict()

    logger.info(f"Running evaluation with params: {eval_params}")

    # Run the evaluation pipeline
    await run_corpus_builder(eval_params, instance_filter=eval_params.get("instance_filter"))
    await run_question_answering(eval_params)
    answers = await run_evaluation(eval_params)
    with open("/data/" + (str)(eval_params.get("name_of_answers_file")), "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    vol.commit()
    if eval_params.get("dashboard"):
        logger.info("Generating dashboard...")
        html = create_dashboard(
            metrics_path=eval_params["metrics_path"],
            aggregate_metrics_path=eval_params["aggregate_metrics_path"],
            output_file=eval_params["dashboard_path"],
            benchmark=eval_params["benchmark"],
        )
        with open("/data/" + (str)(eval_params.get("name_of_html")), "w") as f:
            f.write(html)
        vol.commit()

    # Early return if metrics calculation wasn't requested
    if not eval_params.get("evaluating_answers") or not eval_params.get("calculate_metrics"):
        logger.info(
            "Skipping metrics collection as either evaluating_answers or calculate_metrics is False"
        )
        return None

    logger.info("Everything finished...")

    return True


@app.local_entrypoint()
async def main():
    # List of configurations to run
    configs = [
        EvalConfig(
            task_getter_type="Default",
            benchmark="HotPotQA",
            number_of_samples_in_corpus=24,
            building_corpus_from_scratch=True,
            qa_engine="cognee_graph_completion",
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
            name_of_answers_file="Hotpot_train3.json",
            name_of_html="Hotpot_train3.html",
            instance_filter=[
                "5a8e341c5542995085b373d6",
                "5ab2a308554299340b52553b",
                "5a79c1095542996c55b2dc62",
                "5a8c52685542995e66a475bb",
                "5a734dad5542994cef4bc522",
                "5a74ab6055429916b01641b9",
                "5ae2661d554299495565da60",
                "5a88dcf9554299206df2b383",
                "5ab8179f5542990e739ec817",
                "5a812d0555429938b61422e1",
                "5a79be0e5542994f819ef084",
                "5a875b755542996e4f308796",
                "5ae675245542991bbc9760dc",
                "5ab819065542995dae37ea3c",
                "5a74d64055429916b0164223",
                "5abfea825542994516f45527",
                "5ac279345542990b17b153b0",
                "5ab3c48755429969a97a81b8",
                "5adf35935542993344016c36",
                "5a83d0845542996488c2e4e6",
                "5a7af32e55429931da12c99c",
                "5a7c9ead5542990527d554e4",
                "5ae12aa6554299422ee99617",
                "5a710a915542994082a3e504",
            ],
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=12,
            benchmark="HotPotQA",
            building_corpus_from_scratch=True,
            qa_engine="cognee_graph_completion",
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
            name_of_answers_file="Hotpot_test3.json",
            name_of_html="Hotpot_test3.html",
            instance_filter=[
                "5ae27df25542992decbdcd2a",
                "5a72224755429971e9dc92be",
                "5a8900c75542997e5c09a6ed",
                "5ae1412a55429920d523434c",
                "5ab2342a5542993be8fa98c3",
                "5adde2475542997545bbbdc1",
                "5ac434cb5542997ea680ca2f",
                "5a8aed1755429950cd6afbf1",
                "5ae328f45542991a06ce993c",
                "5ae17f1e5542990adbacf7a6",
                "5ac42f42554299076e296d88",
                "5ab7484c5542992aa3b8c80d",
            ],
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=24,
            building_corpus_from_scratch=True,
            benchmark="TwoWikiMultiHop",
            qa_engine="cognee_graph_completion",
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
            name_of_answers_file="TwoWiki_train3.json",
            name_of_html="TwoWiki_train3.html",
            instance_filter=[
                "37af9394085111ebbd58ac1f6bf848b6",
                "2102541508ac11ebbd82ac1f6bf848b6",
                "b249aa840bdc11eba7f7acde48001122",
                "feb4b9dc0bdb11eba7f7acde48001122",
                "13d3552e0bde11eba7f7acde48001122",
                "cc6c68e4096511ebbdafac1f6bf848b6",
                "10776f4508a211ebbd7aac1f6bf848b6",
                "c096ef9e086d11ebbd62ac1f6bf848b6",
                "20c7b59608db11ebbd9cac1f6bf848b6",
                "7f3724780baf11ebab90acde48001122",
                "482773fc0baf11ebab90acde48001122",
                "e519fa3c0bae11ebab90acde48001122",
                "d956416e086711ebbd5eac1f6bf848b6",
                "89024aba08a411ebbd7dac1f6bf848b6",
                "19a3ad5008c811ebbd91ac1f6bf848b6",
                "ee484526089f11ebbd78ac1f6bf848b6",
                "53625784086511ebbd5eac1f6bf848b6",
                "f02d1c2208b811ebbd88ac1f6bf848b6",
                "a2f105fa088511ebbd6dac1f6bf848b6",
                "52618be00bb011ebab90acde48001122",
                "ec70a8a208a311ebbd7cac1f6bf848b6",
                "42b3c0b80bde11eba7f7acde48001122",
                "c807422a0bda11eba7f7acde48001122",
                "4e7c40ed08ea11ebbda7ac1f6bf848b6",
            ],
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=12,
            building_corpus_from_scratch=True,
            benchmark="TwoWikiMultiHop",
            qa_engine="cognee_graph_completion",
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
            name_of_answers_file="TwoWiki_test3.json",
            name_of_html="TwoWiki_test3.html",
            instance_filter=[
                "5211d89a095011ebbdaeac1f6bf848b6",
                "fe105e54089411ebbd75ac1f6bf848b6",
                "bd6f350408d311ebbd96ac1f6bf848b6",
                "57f2630e08ae11ebbd83ac1f6bf848b6",
                "8d9cf88009b311ebbdb0ac1f6bf848b6",
                "eafb6d960bae11ebab90acde48001122",
                "45153f740bdb11eba7f7acde48001122",
                "385457c20bde11eba7f7acde48001122",
                "45a16d5a0bdb11eba7f7acde48001122",
                "7253afc808c711ebbd91ac1f6bf848b6",
                "d03449820baf11ebab90acde48001122",
                "0ea215140bdd11eba7f7acde48001122",
            ],
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=24,
            building_corpus_from_scratch=True,
            qa_engine="cognee_graph_completion",
            benchmark="Musique",
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
            name_of_answers_file="Musique_train3.json",
            name_of_html="Musique_train3.html",
            instance_filter=[
                "2hop__374495_68633",
                "2hop__735014_83837",
                "2hop__108158_83769",
                "2hop__92051_827343",
                "2hop__55552_158105",
                "2hop__81825_49084",
                "2hop__91667_81007",
                "2hop__696442_51329",
                "3hop1__516535_834494_34099",
                "3hop1__57186_237521_291682",
                "3hop1__475351_160713_77246",
                "3hop2__304722_397371_63959",
                "3hop1__135392_87694_64412",
                "3hop1__354480_834494_33939",
                "3hop1__446612_160545_34751",
                "3hop1__232315_831637_91775",
                "3hop2__222979_132536_40768",
                "3hop2__304722_330033_63959",
                "3hop1__488744_443779_52195",
                "3hop1__146155_131905_41948",
                "4hop1__788226_32392_823060_610794",
                "4hop1__236903_153080_33897_81096",
                "4hop1__199881_378185_282674_759393",
                "4hop1__726391_153080_33952_34109",
            ],
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=12,
            building_corpus_from_scratch=True,
            qa_engine="cognee_graph_completion",
            benchmark="Musique",
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
            name_of_answers_file="Musique_test3.json",
            name_of_html="Musique_test3.html",
            instance_filter=[
                "2hop__272714_113442",
                "2hop__6827_49664",
                "2hop__24648_192417",
                "2hop__85958_87295",
                "3hop2__222979_840908_40768",
                "3hop1__640171_228453_10972",
                "3hop1__92991_78276_68042",
                "3hop1__147162_131905_41948",
                "4hop1__813171_153080_159767_81096",
                "4hop1__726391_153080_33952_33939",
                "4hop1__707078_765799_282674_759393",
                "4hop1__408432_32392_823060_610794",
            ],
        ),
    ]

    # Run evaluations in parallel with different configurations
    modal_tasks = [modal_run_eval2.remote.aio(config.to_dict()) for config in configs]
    await asyncio.gather(*modal_tasks)
