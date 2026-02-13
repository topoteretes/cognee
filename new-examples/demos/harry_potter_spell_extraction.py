import os
import asyncio
import pathlib
from typing import Literal
from pydantic import Field as PydanticField, model_validator

from cognee import config, add, cognify, prune, visualize_graph, search, SearchType
from cognee.low_level import DataPoint
from cognee.modules.engine.utils import generate_node_id


class SpellType(DataPoint):
    """Category node that every extracted spell points to via 'is_a'."""
    name: Literal["Spell"] = "Spell"

    @model_validator(mode="after")
    def set_deterministic_id(self):
        self.id = generate_node_id(self.name)
        return self


class Spell(DataPoint):
    """A magical spell, charm, curse, hex, or jinx from the Harry Potter universe."""
    name: str = PydanticField(
        ...,
        description=(
            "The canonical incantation or common name of the spell "
            "(e.g. 'Alohomora', 'Wingardium Leviosa' , 'Almordis Hotalamus')."
        ),
    )
    description: str = PydanticField(
        default="",
        description=(
            "A short description of what the spell does or its observable effect "
            "as described in the text."
        ),
    )
    is_a: SpellType
    metadata: dict = {"index_fields": ["name"]}

    @model_validator(mode="after")
    def normalize_and_set_id(self):
        n = self.name.strip()
        if n.lower().startswith("the "):
            n = n[4:].strip()
        self.name = n
        if not n:
            self.metadata = {"index_fields": []}
        self.id = generate_node_id(self.name or "__empty__")
        return self


class SpellList(DataPoint):
    """Wrapper allowing zero or many spells per chunk."""
    spells: list[Spell] = []
    metadata: dict = {"index_fields": []}


SPELL_EXTRACTION_PROMPT = """\
You are an expert on the Harry Potter universe. Your task is to extract \
every magical spell mentioned in the text provided.

## What IS a spell (extract these)
- Named incantations with Latin or pseudo-Latin words \
(e.g. "Alohomora", "Wingardium Leviosa", "Petrificus Totalus", "Expelliarmus").
- Named curses, hexes, jinxes, and charms that have a specific incantation \
(e.g. "Avada Kedavra", "Crucio", "Imperio", "Stupefy", "Obliviate").

## What is NOT a spell (do NOT extract these)
- Characters or people (e.g. "Voldemort", "Harry", "Dumbledore", "You-Know-Who").
- Magical objects or artifacts (e.g. "Sorting Hat", "Invisibility Cloak", "Wand", \
"Mirror of Erised", "Philosopher's Stone").
- Potions or draughts (e.g. "Draught of Living Death", "Polyjuice Potion").
- Passwords (e.g. "Caput Draconis", "Pig snout").
- Book titles or chapter names (e.g. "Curses and Countercurses").
- Descriptions of magical effects (e.g. "conjure fire", "cursed broomstick").
- Dialogue, exclamations, or quotes (e.g. "MOTORCYCLES DON'T FLY").
- Locations, creatures, or abstract concepts (e.g. "Hogwarts", "Magic", "Quidditch").

## Extraction rules
1. Extract EVERY distinct spell incantation you find — do not skip any.
2. For the `name` field, use the exact incantation words (e.g. "Wingardium Leviosa").
3. For the `description` field, write one sentence about the spell's effect.
4. If the same spell appears multiple times, extract it only ONCE.
5. If the passage contains NO spells, return an empty `spells` list.
6. Do NOT invent spells not present in the text.
"""


def set_up_config():
    data_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage")).resolve()
    )
    config.data_root_directory(data_directory_path)

    cognee_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system")).resolve()
    )
    config.system_root_directory(cognee_directory_path)


async def visualize_data():
    graph_file_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".artifacts/graph_visualization.html")
        ).resolve()
    )
    await visualize_graph(graph_file_path)



async def main():
    set_up_config()

    await prune.prune_data()
    await prune.prune_system(metadata=True)

    hp_file_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, "01_Harry_Potter.txt")).resolve()
    )

    await add(hp_file_path)
    await cognify(
        graph_model=SpellList,
        custom_prompt=SPELL_EXTRACTION_PROMPT,
    )

    await visualize_data()

    graph_completion = await search(
        query_text="What spells are mentioned in Harry Potter?",
        query_type=SearchType.GRAPH_COMPLETION,
        top_k=100,
    )
    print("\n--- Graph Completion (Spells) ---")
    print(graph_completion)

if __name__ == "__main__":
    asyncio.run(main())
