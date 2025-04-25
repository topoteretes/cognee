import os
import shutil
import cognee
import pathlib
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text_1 = """
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

    text_2 = """
    1. Apple
    Apple is renowned for its innovative consumer electronics and software. Its product lineup includes the iPhone, iPad, Mac computers, and wearables like the Apple Watch. Known for its emphasis on sleek design and user-friendly interfaces, Apple has built a loyal customer base and created a seamless ecosystem that integrates hardware, software, and services.

    2. Google
    Founded in 1998, Google started as a search engine and quickly became the go-to resource for finding information online. Over the years, the company has diversified its offerings to include digital advertising, cloud computing, mobile operating systems (Android), and various web services like Gmail and Google Maps. Google's innovations have played a major role in shaping the internet landscape.

    3. Microsoft
    Microsoft Corporation has been a dominant force in software for decades. Its Windows operating system and Microsoft Office suite are staples in both business and personal computing. In recent years, Microsoft has expanded into cloud computing with Azure, gaming with the Xbox platform, and even hardware through products like the Surface line. This evolution has helped the company maintain its relevance in a rapidly changing tech world.

    4. Amazon
    What began as an online bookstore has grown into one of the largest e-commerce platforms globally. Amazon is known for its vast online marketplace, but its influence extends far beyond retail. With Amazon Web Services (AWS), the company has become a leader in cloud computing, offering robust solutions that power websites, applications, and businesses around the world. Amazon's constant drive for innovation continues to reshape both retail and technology sectors.

    5. Meta
    Meta, originally known as Facebook, revolutionized social media by connecting billions of people worldwide. Beyond its core social networking service, Meta is investing in the next generation of digital experiences through virtual and augmented reality technologies, with projects like Oculus. The company's efforts signal a commitment to evolving digital interaction and building the metaverse—a shared virtual space where users can connect and collaborate.

    Each of these companies has significantly impacted the technology landscape, driving innovation and transforming everyday life through their groundbreaking products and services.
    """

    await cognee.add([text_1, text_2])

    await cognee.cognify()

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) > 10 and len(edges) > 10, "Graph database is not loaded."

    await cognee.delete([text_1, text_2], mode="hard")
    nodes, edges = await graph_engine.get_graph_data()

    assert len(nodes) == 0 and len(edges) == 0, "Document is not deleted."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
