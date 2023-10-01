from typing import Optional

import networkx as nx
from pydantic import BaseModel


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


class DatasetInfo(BaseModel):
    name: str
    text_column: Optional[str] = None
    description: Optional[str] = None
