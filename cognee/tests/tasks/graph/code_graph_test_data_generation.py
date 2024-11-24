import random
import string

import numpy as np

from cognee.shared.CodeGraphEntities import CodeFile, CodeRelationship


def random_str(n, spaces=True):
    candidates = string.ascii_letters + string.digits
    if spaces:
        candidates += "    "
    return "".join(random.choice(candidates) for _ in range(n))


def code_graph_test_data_generation():
    nodes = [
        CodeFile(
            extracted_id=random_str(10, spaces=False),
            type="file",
            source_code=random_str(random.randrange(50, 500)),
        )
        for _ in range(100)
    ]
    n_nodes = len(nodes)
    first_source = np.random.randint(0, n_nodes)
    reached_nodes = {first_source}
    last_iteration = [first_source]
    edges = []
    while len(reached_nodes) < n_nodes:
        for source in last_iteration:
            last_iteration = []
            tries = 0
            while ((len(last_iteration) == 0 or tries < 500)) and (
                len(reached_nodes) < n_nodes
            ):
                tries += 1
                target = np.random.randint(n_nodes)
                if target not in reached_nodes:
                    last_iteration.append(target)
                    edges.append(
                        CodeRelationship(
                            source_id=nodes[source].extracted_id,
                            target_id=nodes[target].extracted_id,
                            type="files",
                            relation="depends",
                        )
                    )
            reached_nodes = reached_nodes.union(set(last_iteration))

    return (nodes, edges)
