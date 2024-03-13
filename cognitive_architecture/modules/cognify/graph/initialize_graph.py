from datetime import datetime
from cognitive_architecture.shared.data_models import DefaultGraphModel, Relationship, UserProperties, UserLocation
from cognitive_architecture.modules.cognify.graph.create import create_semantic_graph

async def initialize_graph(root_id: str):
    graph = DefaultGraphModel(
        id = root_id,
        user_properties = UserProperties(
            custom_properties = {"age": "30"},
            location = UserLocation(
                location_id = "ny",
                description = "New York",
                default_relationship = Relationship(type = "located_in")
            )
        ),
        default_fields = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    await create_semantic_graph(graph)
