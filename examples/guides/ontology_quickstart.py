import asyncio
import cognee
import os
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver


async def main():
    texts = ["Audi produces the R8 and e-tron.", "Apple develops iPhone and MacBook."]

    await cognee.add(texts)
    # or: await cognee.add("/path/to/folder/of/files")

    ontology_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ontology_input_example/basic_ontology.owl"
    )

    # Create full config structure manually
    config: Config = {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
        }
    }

    await cognee.cognify(config=config)


if __name__ == "__main__":
    asyncio.run(main())
