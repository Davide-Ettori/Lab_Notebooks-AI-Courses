g = [
    None,
    [2, 4],
    [1, 3, 4],
    [2, 4, 5],
    [1, 2, 3, 5],
    [3, 4]
]

def get_nodes_degree(graph):
    res = list()
    for node in graph:
        if node is None:
            res.append(None)
        else:
            res.append(len(node))
    return res


def eulerian_cycle(graph):
    degrees = get_nodes_degree(graph)
    for deg in degrees:
        if deg is None:
            continue
        if deg % 2 == 1:
            return False
    return True


def eulerian_path(graph):
    a, b = None, None
    degrees = get_nodes_degree(graph)
    for i, deg in enumerate(degrees):
        if deg is None:
            continue
        if deg % 2 == 1:
            if a is None:
                a = i
            elif b is None:
                b = i
            else:
                return None, None
    if a is None or b is None:
        return None, None
    return a, b

print()
print(eulerian_cycle(g))
print(eulerian_path(g))

def reachable(graph, n):  # BFS
    visited = set()
    queue = list()
    cur = n

    queue.append(cur)

    while len(queue) != 0:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)

        for node in graph[cur]:
            queue.append(node)
    visited.remove(n)
    return list(visited)

print()
print(reachable(g, 2))

def cycle(graph):  # BFS
    visited = set()
    edge = set()

    for i in range(1, len(graph)):  # questo ciclo esterno serve perché ci potrebbero essere parti del grafo sconnesse
        if i in visited:
            continue
        stack = [i]
        while len(stack) > 0:
            cur = stack.pop(0)
            if cur in visited:
                return True
            visited.add(cur)
            for node in graph[cur]:
                if (cur, node) in edge:  # impedisci di ripercorre un arco all'indietro, vuoi un ciclo reale
                    continue
                stack.append(node)
                edge.add((node, cur))
    return False

print()
print(cycle(g))
print(cycle([
    None,
    [1, 2],
    [1, 3],
    [2, 5],
    [1],
    [3]
]))

def is_bridge(graph, edge):  # BFS
    forbidden = [(edge[0], edge[1]), (edge[1], edge[0])]
    stack = [edge[0]]
    visited = set()
    while len(stack) > 0:
        cur = stack.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for node in graph[cur]:
            if (cur, node) in forbidden:
                continue
            stack.append(node)
    return edge[1] not in visited  # se lo hai raggiunto comunque allora l'arco che hai tolto non era un ponte

print()
print(is_bridge(g, (1, 2)))
print(is_bridge([
    None,
    [2],
    [1, 3],
    [2, 4, 5],
    [3, 5],
    [3, 4]
], (2, 3)))