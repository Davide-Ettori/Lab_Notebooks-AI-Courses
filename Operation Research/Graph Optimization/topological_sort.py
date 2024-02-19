import math

# Tutti questi algoritmi valgono sse il grafo è un DAG --> controlla prima

class CycleDetection:
    def __init__(self):
        self.visited = set()
        self.top_sort = list()

    def dfs(self, graph, cur):
        if cur in self.visited:
            return
        self.visited.add(cur)
        for node in graph[cur]:
            self.dfs(graph, node)
        self.top_sort.append(cur)

    def is_cyclic(self, graph):
        for i in range(len(graph)):  # faccio la topological search
            if i in self.visited:
                continue
            self.dfs(graph, i)
        self.top_sort.reverse()
        idx_map = dict()
        for i, n in enumerate(self.top_sort):  # mappo a ogni nodo il suo indice
            idx_map[n] = i

        for node in range(len(graph)):  # controllo se la topological search ha avuto successo, ovvero DAG (niente cicli)
            for edge in graph[node]:
                if idx_map[edge] <= idx_map[node]:
                    return True
        return False

class ShortestPath:
    def __init__(self):
        self.visited = set()
        self.top_sort = list()

    def dfs(self, graph, cur):
        if cur in self.visited:
            return
        self.visited.add(cur)
        for node in graph[cur]:
            self.dfs(graph, node)
        self.top_sort.append(cur)

    def shortest_path(self, graph, weights, s):  # funziona anche con pesi negativi
        self.dfs(graph, s)  # mi assicuro che questo nodo sia il primo della topological sort
        for i in range(len(graph)):
            if i in self.visited:
                continue
            self.dfs(graph, i)
        self.top_sort.reverse()

        dist = [math.inf if i != 0 else 0 for i in range(len(graph))]

        for node in self.top_sort:
            for v in graph[node]:
                w = weights[(node, v)]
                dist[v] = min(dist[v], dist[node] + w)

        return dist  # percorso migliore da S a tutti gli altri nodi

class HamiltonianPath:
    def __init__(self):
        self.visited = set()
        self.top_sort = list()

    def dfs(self, graph, cur):
        if cur in self.visited:
            return
        self.visited.add(cur)
        for node in graph[cur]:
            self.dfs(graph, node)
        self.top_sort.append(cur)

    def hamiltonian_path(self, graph):
        for i in range(len(graph)):
            if i in self.visited:
                continue
            self.dfs(graph, i)
        self.top_sort.reverse()

        edges = set()
        for i in range(len(graph)):
            for j in graph[i]:
                edges.add((i, j))

        for i in range(1, len(self.top_sort)):
            if (self.top_sort[i - 1], self.top_sort[i]) not in edges:
                return None
        return self.top_sort

class SchedulingOptimization:
    def __init__(self):
        self.visited = set()
        self.top_sort = list()

    def dfs(self, graph, cur):
        if cur in self.visited:
            return
        self.visited.add(cur)
        for node in graph[cur]:
            self.dfs(graph, node)
        self.top_sort.append(cur)

    def topological_sort(self, graph):
        for i in range(len(graph)):
            if i in self.visited:
                continue
            self.dfs(graph, i)
        self.top_sort.reverse()
        return self.top_sort

# T(N) = O(|V| + |E|), S(N) = O(|V|), K = O(25)