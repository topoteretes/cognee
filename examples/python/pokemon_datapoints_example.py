# Standard library imports
import os
import json
import asyncio
import pathlib
from uuid import uuid5, NAMESPACE_OID
from typing import List, Optional
from pathlib import Path

import dlt
import requests
import cognee
from cognee.low_level import DataPoint, setup as cognee_setup
from cognee.api.v1.search import SearchType
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.pipelines import run_tasks


BASE_URL = "https://pokeapi.co/api/v2/"
os.environ["BUCKET_URL"] = "./.data_storage"
os.environ["DATA_WRITER__DISABLE_COMPRESSION"] = "true"

# Data Models
class Abilities(DataPoint):
    name: str = "Abilities"
    metadata: dict = {"index_fields": ["name"]}

class PokemonAbility(DataPoint):
    name: str
    ability__name: str
    ability__url: str
    is_hidden: bool
    slot: int
    _dlt_load_id: str
    _dlt_id: str
    _dlt_parent_id: str
    _dlt_list_idx: str
    is_type: Abilities
    metadata: dict = {"index_fields": ["ability__name"]}

class Pokemons(DataPoint):
    name: str = "Pokemons"
    have: Abilities
    metadata: dict = {"index_fields": ["name"]}

class Pokemon(DataPoint):
    name: str
    base_experience: int
    height: int
    weight: int
    is_default: bool
    order: int
    location_area_encounters: str
    species__name: str
    species__url: str
    cries__latest: str
    cries__legacy: str
    sprites__front_default: str
    sprites__front_shiny: str
    sprites__back_default: Optional[str]
    sprites__back_shiny: Optional[str]
    _dlt_load_id: str
    _dlt_id: str
    is_type: Pokemons
    abilities: List[PokemonAbility]
    metadata: dict = {"index_fields": ["name"]}

# Data Collection Functions
@dlt.resource(write_disposition="replace")
def pokemon_list(limit: int = 50):
    response = requests.get(f"{BASE_URL}pokemon", params={"limit": limit})
    response.raise_for_status()
    yield response.json()["results"]

@dlt.transformer(data_from=pokemon_list)
def pokemon_details(pokemons):
    """Fetches detailed info for each Pokémon"""
    for pokemon in pokemons:
        response = requests.get(pokemon["url"])
        response.raise_for_status()
        yield response.json()

# Data Loading Functions
def load_abilities_data(jsonl_abilities):
    abilities_root = Abilities()
    pokemon_abilities = []

    for jsonl_ability in jsonl_abilities:
        with open(jsonl_ability, "r") as f:
            for line in f:
                ability = json.loads(line)
                ability["id"] = uuid5(NAMESPACE_OID, ability["_dlt_id"])
                ability["name"] = ability["ability__name"]
                ability["is_type"] = abilities_root
                pokemon_abilities.append(ability)

    return abilities_root, pokemon_abilities

def load_pokemon_data(jsonl_pokemons, pokemon_abilities, pokemon_root):
    pokemons = []

    for jsonl_pokemon in jsonl_pokemons:
        with open(jsonl_pokemon, "r") as f:
            for line in f:
                pokemon_data = json.loads(line)
                abilities = [
                    ability for ability in pokemon_abilities
                    if ability["_dlt_parent_id"] == pokemon_data["_dlt_id"]
                ]
                pokemon_data["external_id"] = pokemon_data["id"]
                pokemon_data["id"] = uuid5(NAMESPACE_OID, str(pokemon_data["id"]))
                pokemon_data["abilities"] = [PokemonAbility(**ability) for ability in abilities]
                pokemon_data["is_type"] = pokemon_root
                pokemons.append(Pokemon(**pokemon_data))

    return pokemons

# Main Application Logic
async def setup_and_process_data():
    """Setup configuration and process Pokemon data"""
    # Setup configuration
    data_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage")).resolve())
    cognee_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system")).resolve())

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # Initialize pipeline and collect data
    pipeline = dlt.pipeline(
        pipeline_name="pokemon_pipeline",
        destination="filesystem",
        dataset_name="pokemon_data",
    )
    info = pipeline.run([pokemon_list, pokemon_details])
    print(info)

    # Load and process data
    STORAGE_PATH = Path(".data_storage/pokemon_data/pokemon_details")
    jsonl_pokemons = sorted(STORAGE_PATH.glob("*.jsonl"))
    if not jsonl_pokemons:
        raise FileNotFoundError("No JSONL files found in the storage directory.")

    ABILITIES_PATH = Path(".data_storage/pokemon_data/pokemon_details__abilities")
    jsonl_abilities = sorted(ABILITIES_PATH.glob("*.jsonl"))
    if not jsonl_abilities:
        raise FileNotFoundError("No JSONL files found in the storage directory.")

    # Process data
    abilities_root, pokemon_abilities = load_abilities_data(jsonl_abilities)
    pokemon_root = Pokemons(have=abilities_root)
    pokemons = load_pokemon_data(jsonl_pokemons, pokemon_abilities, pokemon_root)

    return pokemons

async def pokemon_cognify(pokemons):
    """Process Pokemon data with Cognee and perform search"""
    # Setup and run Cognee tasks
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee_setup()

    tasks = [Task(add_data_points, task_config={"batch_size": 50})]
    results = run_tasks(
        tasks=tasks,
        data=pokemons,
        dataset_id=uuid5(NAMESPACE_OID, "Pokemon"),
        pipeline_name='pokemon_pipeline',
    )

    async for result in results:
        print(result)
    print("Done")

    # Perform search
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="pokemons?"
    )

    print("Search results:")
    for result_text in search_results:
        print(result_text)

async def main():
    pokemons = await setup_and_process_data()
    await pokemon_cognify(pokemons)

if __name__ == "__main__":
    asyncio.run(main())
