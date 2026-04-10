from __future__ import annotations
import asyncio
import json

from typing import List

import cognee

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.graph_model_utils import (
    graph_model_to_graph_schema,
    graph_schema_to_graph_model,
)

from pydantic import BaseModel


# Define a custom graph model for programming languages.
class FieldType(BaseModel):
    name: str = "Field"


class Field(BaseModel):
    name: str
    is_type: FieldType
    # metadata: dict = {"index_fields": ["name"]}


class ProgrammingLanguageType(BaseModel):
    name: str = "Programming Language"


class ProgrammingLanguage(BaseModel):
    name: str
    used_in: list[Field] = []
    is_type: ProgrammingLanguageType
    # metadata: dict = {"index_fields": ["name"]}


class ShopType(DataPoint):
    name: str


class ProfessionType(DataPoint):
    name: str


class Employee(DataPoint):
    name: str
    is_a: ProfessionType


class CraftShop(DataPoint):
    name: str
    is_type: ShopType
    employs: List[Employee]


class Town(DataPoint):
    name: str
    contains: List[CraftShop]


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    schema_dict = graph_model_to_graph_schema(Town)
    graph_model_schema_json = json.dumps(schema_dict)

    # model_schema = build_schema_without_base(Town, DataPoint)
    # print(json.dumps(model_schema))

    graph_model = graph_schema_to_graph_model(schema_dict)

    print(graph_model_schema_json)

    user_prompt = render_prompt(
        "custom_prompt_generation_user.txt", {"GRAPH_SCHEMA_JSON": graph_model_schema_json}
    )
    system_prompt = render_prompt("custom_prompt_generation_system.txt", {})

    custom_prompt = await LLMGateway.acreate_structured_output(
        text_input=user_prompt, system_prompt=system_prompt, response_model=str
    )
    print("Custom prompt generation complete")
    print(custom_prompt)

    # await cognee.add("""
    #     Python is a programming language widely used in machine learning, data analysis, and web development.
    #     Rust is a programming language used in systems programming, embedded software, and cybersecurity.
    #     SQL is a programming language used in database management and business intelligence reporting.
    #     JavaScript is a programming language used in frontend web development and interactive user interface design.
    #     Go is a programming language used in cloud infrastructure and backend microservices.
    #     R is a programming language used in statistics and academic research.
    # """)

    await cognee.add("""
    In Novi Sad, where the Danube moved like a patient storyteller and the old streets still remembered every pair of footsteps, mornings began with shutters opening one by one along Dunavska Street. The town’s craft quarter was small, but everyone knew it as the heart of honest work: thread, leather, silver, and the quiet pride of hands that knew their trade better than any machine.

At the corner stood Golden Needle Tailor Shop, a narrow place with tall windows and warm yellow light. Inside worked two tailors who could turn cloth into confidence. Milan Vuković, aged 34, was known for sharp suits and sharper jokes, always humming old tamburica songs while pinning sleeves in place. Beside him worked Luka Savić, aged 27, who had a calmer touch and a gift for listening; people came in asking for a jacket and left feeling understood. They measured carefully, argued kindly about lapel widths, and finished each day brushing chalk from their fingers like bakers dusting flour.

A little farther down the street was Danube Step Shoemaking Shop, where the smell of leather and polish wrapped around visitors the moment they entered. Two shoemakers kept the place alive. Petar Kostić, aged 41, had broad hands and the patience to rebuild a boot sole as if restoring a family heirloom. His partner, Stefan Ilić, aged 29, specialized in elegant city shoes and had a reputation for making even practical footwear look dignified. They worked side by side at scarred wooden benches, tapping nails in steady rhythm, telling stories about customers who wore their shoes to weddings, first jobs, and long-awaited reunions.

Near Liberty Square, where sunlight caught every shop sign, stood Silver Bloom Jewelery Shop. It was the brightest of the three, with small lamps aimed at glass cases where rings and pendants sparkled like captured rain. Two jewelers worked there. Ana Radić, aged 32, designed delicate pieces inspired by river waves and willow leaves, and many in town wore her work on special days. With her was Jelena Marković, aged 45, who had spent decades repairing heirloom brooches and resizing wedding rings, always treating each piece as if it carried a secret. She could look at an old chain and tell, almost exactly, how many birthdays it had witnessed.

Though each shop had its own craft, they were all connected through the same spirit and the same town office records, each listed proudly as having its office in Novi Sad. They shared tools when needed, sent customers to one another, and met every Thursday evening at a tiny kafana nearby. Milan once mended Petar’s torn apron. Petar repaired Luka’s favorite boots for free after a rainy winter. Ana made a silver thimble for Milan’s birthday, and Jelena engraved Stefan’s initials on a shoehorn “so you stop losing it every month.”

One spring, the town announced a festival celebrating local artisans. At first, the three shops worked separately. Then someone suggested they create something together: an outfit, shoes, and jewelry that told a single story of Novi Sad. Golden Needle designed a deep blue coat lined with fine stitching in the pattern of Danube currents. Danube Step crafted polished black shoes with hand-cut details shaped like fortress stones from Petrovaradin. Silver Bloom added a silver pin and cufflinks, each engraved with tiny swallows in flight.

When the finished set was shown in the square, people fell silent for a moment before applauding. It wasn’t only beautiful because of the craftsmanship. It was beautiful because it carried six signatures hidden in seams, soles, and clasps: Milan, Luka, Petar, Stefan, Ana, and Jelena. Six people, three shops, one town.

That evening, as lights reflected in the river and Novi Sad settled into its gentle night sounds, the craft quarter closed its doors one by one. But behind those doors remained the same steady promise as always: if something was torn, worn, or broken, someone in town knew how to mend it.
""")

    await cognee.cognify(
        graph_model=graph_model,
        custom_prompt=custom_prompt,
    )

    await cognee.visualize_graph()

    result = await cognee.search("Is there a tailor in Novi Sad?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
