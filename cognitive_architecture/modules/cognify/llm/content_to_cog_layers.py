from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client


async def content_to_cog_layers(memory_name: str, payload: list):
    llm_client = get_llm_client()

    # data_points = list()
    # for point in map(create_data_point, payload):
    #     data_points.append(await point)

    return await llm_client.acreate_structured_output(memory_name, payload, model="text-embedding-ada-002")




