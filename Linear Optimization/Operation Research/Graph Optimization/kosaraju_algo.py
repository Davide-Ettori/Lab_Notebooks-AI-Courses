class Graph:
    def __init__(self, vertex):
        self.visited = set()
        self.stack = list()
        self.res = list()

    def dfs_and_append(self, graph, cur):
        if cur in self.visited:
            return
        self.visited.add(cur)
        for node in graph[cur]:
            self.dfs_and_append(graph, node)
        self.stack.append(cur)  # metto nella stack alla fine --> come in topological sort

    def dfs(self, graph, cur):
        if cur in self.visited:
            return
        self.visited.add(cur)
        self.res[-1].add(cur)  # aggiungo i nodi al ultimo SCC trovato
        for node in graph[cur]:
            self.dfs(graph, node)

    @staticmethod
    def transpose(graph):
        new_graph = [list() for _ in range(len(graph))]
        for i in range(len(graph)):
            for node in graph[i]:
                # noinspection PyTypeChecker
                new_graph[node].append(i)  # ignora questo warning --> tipico di PyCharm
        return new_graph

    def find_SCC(self, graph):
        for i in range(len(graph)):
            if i in self.visited:
                continue
            self.dfs_and_append(graph, i)

        graph_trans = Graph.transpose(graph)
        self.visited = set()

        while len(self.stack) > 0:
            node = self.stack.pop()
            if node in self.visited:
                continue
            self.res.append(set())
            self.dfs(graph_trans, node)

        return self.res

# complessità uguale alla classica DFS