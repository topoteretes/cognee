from cognee.modules.engine.utils import generate_node_id


def filter_overlapping_relationships(*relationship_groups):
    relationship_count = {}
    overlapping_relationships = []

    for group in relationship_groups:
        for relationship in group:
            relationship_id = f"{relationship[0]}_{relationship[2]}_{relationship[1]}"

            if relationship_id not in relationship_count:
                relationship_count[relationship_id] = 1
            else:
                relationship_count[relationship_id] += 1

    index = 0
    grouped_relationships = []
    for group in relationship_groups:
        grouped_relationships.append([])

        for relationship in group:
            relationship_id = f"{relationship[0]}_{relationship[2]}_{relationship[1]}"

            if relationship_count[relationship_id] == 1:
                grouped_relationships[index].append(relationship)
            else:
                if relationship not in overlapping_relationships:
                    overlapping_relationships.append(relationship)

        index += 1

    return overlapping_relationships, *grouped_relationships
