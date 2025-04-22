import asyncio
import cognee
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.api.v1.search import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines import run_tasks, Task
from cognee.tasks.experimental_tasks.node_set_edge_association import node_set_edge_association

text_a = """
    Leading financial technology firms like Stripe, Square, and Revolut are redefining digital commerce by embedding AI
    into their payment ecosystems. Stripe leverages machine learning to detect and prevent fraud in real time,
    while Square uses predictive analytics to offer customized lending solutions to small businesses.
    Meanwhile, Revolut applies AI algorithms to automate wealth management services, enabling users to invest,
    save, and budget with unparalleled personalization and efficiency.
    """

text_b = """
    Pioneering AI companies such as OpenAI, Anthropic, and DeepMind are advancing self-supervised
    learning techniques that empower systems to autonomously evolve their cognitive capabilities.
    OpenAI's models interpret complex multimodal data with minimal human annotation, while Anthropic’s
    Constitutional AI approach refines alignment and safety. DeepMind continues to push boundaries with
    breakthroughs like AlphaFold, illustrating the power of AI to decipher intricate biological structures
    without exhaustive manual input.
    """

text_c = """
    MedTech innovators like Medtronic, Butterfly Network, and Intuitive Surgical are revolutionizing
    healthcare delivery through smart devices and AI-driven platforms. Medtronic's connected insulin
    pumps enable real-time glucose monitoring, Butterfly Network’s portable ultrasound devices bring
    diagnostic imaging to remote areas, and Intuitive Surgical’s da Vinci system enhances precision
    in minimally invasive surgeries. Together, these companies are reshaping clinical pathways and
    extending care beyond traditional hospital settings.
    """

node_set_a = ["AI_NODESET", "FinTech_NODESET"]
node_set_b = ["AI_NODESET"]
node_set_c = ["MedTech_NODESET"]


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(text_a, node_set=node_set_a)
    await cognee.add(text_b, node_set=node_set_b)
    await cognee.add(text_c, node_set=node_set_c)
    await cognee.cognify()

    tasks = [Task(node_set_edge_association)]

    user = await get_default_user()
    pipeline = run_tasks(tasks=tasks, user=user)

    async for pipeline_status in pipeline:
        print(f"Pipeline run status: {pipeline_status.pipeline_name} - {pipeline_status.status}")

    print()


if __name__ == "__main__":
    logger = get_logger(level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.run(main())
