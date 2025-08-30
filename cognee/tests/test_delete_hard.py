import os
import shutil
import cognee
import pathlib
from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods import get_dataset_data

logger = get_logger()


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    pdf_document = os.path.join(
        pathlib.Path(__file__).parent, "test_data/artificial-intelligence.pdf"
    )

    txt_document = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing_copy.txt"
    )

    audio_document = os.path.join(pathlib.Path(__file__).parent, "test_data/text_to_speech.mp3")

    image_document = os.path.join(pathlib.Path(__file__).parent, "test_data/example.png")

    unstructured_document = os.path.join(pathlib.Path(__file__).parent, "test_data/example.pptx")

    text_document_as_literal = """
    1. Audi
    Audi is known for its modern designs and advanced technology. Founded in the early 1900s, the brand has earned a reputation for precision engineering and innovation. With features like the Quattro all-wheel-drive system, Audi offers a range of vehicles from stylish sedans to high-performance sports cars.

    2. BMW
    BMW, short for Bayerische Motoren Werke, is celebrated for its focus on performance and driving pleasure. The company's vehicles are designed to provide a dynamic and engaging driving experience, and their slogan, "The Ultimate Driving Machine," reflects that commitment. BMW produces a variety of cars that combine luxury with sporty performance.

    3. Mercedes-Benz
    Mercedes-Benz is synonymous with luxury and quality. With a history dating back to the early 20th century, the brand is known for its elegant designs, innovative safety features, and high-quality engineering. Mercedes-Benz manufactures not only luxury sedans but also SUVs, sports cars, and commercial vehicles, catering to a wide range of needs.

    4. Porsche
    Porsche is a name that stands for high-performance sports cars. Founded in 1931, the brand has become famous for models like the iconic Porsche 911. Porsche cars are celebrated for their speed, precision, and distinctive design, appealing to car enthusiasts who value both performance and style.

    5. Volkswagen
    Volkswagen, which means "people's car" in German, was established with the idea of making affordable and reliable vehicles accessible to everyone. Over the years, Volkswagen has produced several iconic models, such as the Beetle and the Golf. Today, it remains one of the largest car manufacturers in the world, offering a wide range of vehicles that balance practicality with quality.

    Each of these car manufacturer contributes to Germany's reputation as a leader in the global automotive industry, showcasing a blend of innovation, performance, and design excellence.
    """

    # Add documents and get dataset information
    add_result = await cognee.add(
        [
            pdf_document,
            txt_document,
            text_document_as_literal,
            unstructured_document,
            audio_document,
            image_document,
        ]
    )
    dataset_id = add_result.dataset_id

    await cognee.cognify()

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) > 10 and len(edges) > 10, "Graph database is not loaded."

    # Get the data IDs from the dataset
    dataset_data = await get_dataset_data(dataset_id)
    assert len(dataset_data) > 0, "Dataset should contain data"

    # Delete each document using its ID
    for data_item in dataset_data:
        await cognee.delete(data_item.id, dataset_id, mode="hard")

    nodes, edges = await graph_engine.get_graph_data()

    assert len(nodes) == 0 and len(edges) == 0, "Document is not deleted with hard delete."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
