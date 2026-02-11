def filter_overlapping_entities(*entity_groups):
    entity_count = {}
    overlapping_entities = []

    for group in entity_groups:
        for entity in group:
            if entity.id not in entity_count:
                entity_count[entity.id] = 1
            else:
                entity_count[entity.id] += 1

    index = 0
    grouped_entities = []
    for group in entity_groups:
        grouped_entities.append([])

        for entity in group:
            if entity_count[entity.id] == 1:
                grouped_entities[index].append(entity)
            else:
                if entity not in overlapping_entities:
                    overlapping_entities.append(entity)

        index += 1

    return overlapping_entities, *grouped_entities
