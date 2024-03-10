
def fetch_context(CONNECTED_GRAPH, id):
    relevant_context = []
    for n,attr in CONNECTED_GRAPH.nodes(data=True):
        if id in n:
            for n_, attr_ in CONNECTED_GRAPH.nodes(data=True):
                relevant_layer = attr['layer_uuid']

                if attr_.get('layer_uuid') == relevant_layer:
                    print(attr_['description'])
                    relevant_context.append(attr_['description'])

    return relevant_context
