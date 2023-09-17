import networkx as nx


def weak_topological_sort(g: nx.DiGraph, nodes):
    sccs = list(nx.strongly_connected_components(g))
    condensed = nx.condensation(g, scc=sccs)

    result = []
    for scc_index in nx.topological_sort(condensed):
        scc = sccs[scc_index]
        for node in scc:
            if node in nodes:
                result.append(node)

    return result
