from datetime import datetime
from cognee.shared.data_models import DefaultGraphModel, Relationship, UserProperties, UserLocation
from cognee.modules.cognify.graph.create import create_semantic_graph

async def initialize_graph(root_id: str, graphdatamodel, graph_client):
    if graphdatamodel:
        graph = graphdatamodel(id = root_id)
        graph_ = await create_semantic_graph(graph, graph_client)
        return graph_
    else:
        print("Creating default graph")

        graph = DefaultGraphModel(
            node_id = root_id,
            user_properties = UserProperties(
                custom_properties = { "age": "30" },
                location = UserLocation(
                    location_id = "ny",
                    description = "New York",
                    default_relationship = Relationship(
                        type = "located_in",
                        source = "UserProperties",
                        target = "ny",
                    )
                ),
                default_relationship = Relationship(
                    type = "has_properties",
                    source = root_id,
                    target = "UserProperties",
                )
            ),
            default_relationship = Relationship(
                type = "has_properties",
                source = root_id,
                target = "UserProperties"
            ),
            default_fields = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        return await create_semantic_graph(graph, graph_client)
