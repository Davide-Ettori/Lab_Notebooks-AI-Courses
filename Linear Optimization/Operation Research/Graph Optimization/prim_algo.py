import math

g = [
    None,
    [(2, 5), (4, 1)],
    [(3, 3), (4, 2), (1, 5)],
    [(2, 3), (5, 5), (4, 2)],
    [(2, 2), (3, 2), (1, 1), (5, 1)],
    [(3, 5), (4, 1)]
]

def min_span_tree_prim(graph):  # Prim Algo --> Suppose non-negative weights
    if len(graph) <= 2:
        return None, None

    res = list()
    visited = set()
    tot = 0

    visited.add(1)
    while len(visited) != len(graph) - 1:
        min_node = None
        start_node = None
        min_cost = math.inf
        for vis in visited:
            for node, cost in graph[vis]:
                if node in visited:
                    continue
                if cost < min_cost:
                    min_cost = cost
                    min_node = node
                    start_node = vis
        tot += min_cost
        res.append((start_node, min_node))
        visited.add(min_node)
    return res, tot

print()
print(min_span_tree_prim(g)[0], " --> ", min_span_tree_prim(g)[1])  # T(N) = O(n^2), S(N) = O(N), K = O(25)