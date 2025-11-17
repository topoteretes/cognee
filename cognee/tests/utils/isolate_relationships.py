def isolate_relationships(source_relationships, *other_relationships):
    final_relationships = []
    cache = {relationship[2]: 1 for relationship in source_relationships}
    duplicated_relationships = {}

    for relationships in other_relationships:
        for relationship in relationships:
            if relationship[2] not in cache:
                cache[relationship[2]] = 0

            cache[relationship[2]] += 1

            if cache[relationship[2]] == 2:
                duplicated_relationships[relationship[2]] = True

    for relationship in source_relationships:
        if relationship[2] not in duplicated_relationships:
            final_relationships.append(relationship)

    return final_relationships
