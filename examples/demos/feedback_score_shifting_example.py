import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import ERROR, setup_logging

cache_config = get_cache_config()
if not cache_config.caching or cache_config.cache_backend != "fs":
    raise CogneeConfigurationError(
        "feedback_score_shifting_example requires caching=True and CACHE_BACKEND=fs."
    )

TEXT_1 = """
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

TEXT_2 = """
1. Apple
Apple is renowned for its innovative consumer electronics and software. Its product lineup includes the iPhone, iPad, Mac computers, and wearables like the Apple Watch. Known for its emphasis on sleek design and user-friendly interfaces, Apple has built a loyal customer base and created a seamless ecosystem that integrates hardware, software, and services.

2. Google
Founded in 1998, Google started as a search engine and quickly became the go-to resource for finding information online. Over the years, the company has diversified its offerings to include digital advertising, cloud computing, mobile operating systems (Android), and various web services like Gmail and Google Maps. Google's innovations have played a major role in shaping the internet landscape.

3. Microsoft
Microsoft Corporation has been a dominant force in software for decades. Its Windows operating system and Microsoft Office suite are staples in both business and personal computing. In recent years, Microsoft has expanded into cloud computing with Azure, gaming with the Xbox platform, and even hardware through products like the Surface line. This evolution has helped the company maintain its relevance in a rapidly changing tech world.

4. Amazon
What began as an online bookstore has grown into one of the largest e-commerce platforms globally. Amazon is known for its vast online marketplace, but its influence extends far beyond retail. With Amazon Web Services (AWS), the company has become a leader in cloud computing, offering robust solutions that power websites, applications, and businesses around the world. Amazon's constant drive for innovation continues to reshape both retail and technology sectors.

5. Meta
Meta, originally known as Facebook, revolutionized social media by connecting billions of people worldwide. Beyond its core social networking service, Meta is investing in the next generation of digital experiences through virtual and augmented reality technologies, with projects like Oculus. The company's efforts signal a commitment to evolving digital interaction and building the metaverse-a shared virtual space where users can connect and collaborate.

Each of these companies has significantly impacted the technology landscape, driving innovation and transforming everyday life through their groundbreaking products and services.
"""


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add([TEXT_1, TEXT_2])
    await cognee.cognify()

    user = await get_default_user()
    session_id = "feedback_influence_minimal_demo"

    print("Step 1: Ask cars-specific question and give positive feedback (5).")
    await cognee.search(
        query_text="Which German car manufacturers are described and what are they known for?",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        session_id=session_id,
    )
    qa_cars = (await cognee.session.get_session(session_id=session_id, user=user, last_n=1))[0]
    await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=qa_cars.qa_id,
        feedback_score=5,
        feedback_text="Cars-focused context is exactly what I want.",
        user=user,
    )
    print("  Added feedback score=5 for cars context.\n")

    print("Step 2: Ask companies-specific question and give negative feedback (1).")
    await cognee.search(
        query_text="Which technology companies are described and what are their products?",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        session_id=session_id,
    )
    qa_companies = (await cognee.session.get_session(session_id=session_id, user=user, last_n=1))[0]
    await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=qa_companies.qa_id,
        feedback_score=1,
        feedback_text="Companies-focused context is less useful for this goal.",
        user=user,
    )
    print("  Added feedback score=1 for companies context.\n")

    print("Step 3: Apply feedback into graph feedback_weight values (memify).")
    await apply_feedback_weights_pipeline(user=user, session_ids=[session_id], alpha=0.9)
    print("  Feedback weights applied.\n")

    print("Step 4: Ask one neutral query while sweeping beta.")
    print(
        "  As beta increases, ranking should shift toward positively-rated context (companies focused on car manufacturers)."
        " 1 means only feedback score is taken into account nothing else.\n"
    )
    final_query = "List the companies in the context"
    for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        answer = await cognee.search(
            query_text=final_query,
            query_type=SearchType.GRAPH_COMPLETION,
            user=user,
            feedback_influence=beta,
        )
        print(f"\n--- beta = {beta:.1f} ({beta * 100:.0f}% feedback influence) ---")
        print(str(answer))


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
