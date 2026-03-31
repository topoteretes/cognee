from __future__ import annotations
import asyncio
import json

from typing import List

import cognee
import difflib

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.graph_model_utils import graph_model_to_graph_schema, graph_schema_to_graph_model

from copy import deepcopy
from typing import Any, Type
from pydantic import BaseModel


def build_schema_without_base(
    model: Type[BaseModel],
    base: Type[BaseModel],
) -> dict[str, Any]:
    """
    Generate JSON schema for `model`, remove inherited `base` fields,
    prune base defs, and repair/remove broken local $refs.
    """
    schema = deepcopy(model.model_json_schema())
    base_fields = set(base.model_fields.keys())
    base_name = base.__name__

    # 1) Remove base fields from every object schema (root + $defs)
    def strip_base_fields(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for f in list(props.keys()):
                    if f in base_fields:
                        props.pop(f, None)
                if "required" in node and isinstance(node["required"], list):
                    node["required"] = [r for r in node["required"] if r in props]
            for v in node.values():
                strip_base_fields(v)
        elif isinstance(node, list):
            for item in node:
                strip_base_fields(item)

    strip_base_fields(schema)

    # 2) Prune base-like defs
    defs = schema.get("$defs", {})
    if isinstance(defs, dict):
        to_drop = set()
        for def_name, def_schema in defs.items():
            title = def_schema.get("title") if isinstance(def_schema, dict) else None
            if title == base_name or base_name in def_name:
                to_drop.add(def_name)
        for d in to_drop:
            defs.pop(d, None)

    # 3) Remove/repair broken local refs after pruning defs
    existing_defs = set(schema.get("$defs", {}).keys())

    def ref_to_def_name(ref: str) -> str | None:
        pfx = "#/$defs/"
        return ref[len(pfx) :] if ref.startswith(pfx) else None

    def prune_broken(node: Any) -> Any:
        if isinstance(node, dict):
            # direct local ref
            if "$ref" in node and isinstance(node["$ref"], str):
                dname = ref_to_def_name(node["$ref"])
                if dname is not None and dname not in existing_defs:
                    return None  # drop this schema branch

            out = {}
            for k, v in node.items():
                if k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                    pruned = [prune_broken(x) for x in v]
                    pruned = [x for x in pruned if x is not None]
                    if pruned:
                        out[k] = pruned
                    else:
                        return None
                elif k == "properties" and isinstance(v, dict):
                    new_props = {}
                    for pn, ps in v.items():
                        fixed = prune_broken(ps)
                        if fixed is not None:
                            new_props[pn] = fixed
                    out[k] = new_props
                else:
                    out[k] = prune_broken(v)

            # keep required aligned with remaining properties
            if "properties" in out and "required" in out and isinstance(out["required"], list):
                prop_names = set(out["properties"].keys())
                out["required"] = [r for r in out["required"] if r in prop_names]

            return out

        if isinstance(node, list):
            pruned = [prune_broken(x) for x in node]
            return [x for x in pruned if x is not None]

        return node

    fixed = prune_broken(schema)
    return fixed if isinstance(fixed, dict) else {}


# Define a custom graph model for programming languages.
class FieldType(DataPoint):
    name: str = "Field"


class Field(DataPoint):
    name: str
    is_type: FieldType
    metadata: dict = {"index_fields": ["name"]}


class ProgrammingLanguageType(DataPoint):
    name: str = "Programming Language"


class ProgrammingLanguage(DataPoint):
    name: str
    used_in: list[Field] = []
    is_type: ProgrammingLanguageType
    metadata: dict = {"index_fields": ["name"]}


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

    # schema_dict = graph_model_to_graph_schema(Town)
    # graph_model_schema_json = json.dumps(schema_dict)

    # print(graph_model_schema_json)

    model_schema = build_schema_without_base(Town, DataPoint)
    # print(json.dumps(model_schema))

    graph_model = graph_schema_to_graph_model(model_schema)

    user_prompt = render_prompt(
        "custom_prompt_generation_user.txt", {"GRAPH_SCHEMA_JSON": model_schema}
    )
    system_prompt = render_prompt("custom_prompt_generation_system.txt", {})

    custom_prompt = await LLMGateway.acreate_structured_output(
        text_input=user_prompt, system_prompt=system_prompt, response_model=str
    )

    print("Custom prompt generation complete")

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

    print(custom_prompt)

    await cognee.visualize_graph()

    result = await cognee.search("Is there a tailor in Novi Sad?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
