import asyncio

import cognee


async def main():
    dataset_name = "graph_report_example"
    await cognee.add(
        [
            "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.",
            "The Analytical Engine influenced later work on general-purpose computers.",
        ],
        dataset_name=dataset_name,
        node_set=["computing_history"],
    )
    await cognee.cognify(datasets=[dataset_name])

    report = await cognee.report(
        datasets=[dataset_name],
        destination_file_path="graph_report.md",
    )
    print(report["hub_nodes"])

    search_results = await cognee.search(
        query_text="Create a graph insight report",
        query_type=cognee.SearchType.GRAPH_REPORT,
        datasets=[dataset_name],
        retriever_specific_config={"use_llm_questions": False},
    )
    print(search_results)


if __name__ == "__main__":
    asyncio.run(main())
