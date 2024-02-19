g = [
    [1],
    [0, 2, 4],
    [1, 3, 4],
    [2, 4, 5],
    [1, 2, 3, 5],
    [3, 4]
]

class Solution:
    def __init__(self):
        self.visited = set()

    def dfs(self, graph, cur, step):
        if step < 0:
            return
        if cur in self.visited:
            return
        self.visited.add(cur)
        for node in graph[cur]:
            self.dfs(graph, node, step - 1)

    def reachable_in_n_steps(self, graph, s, step):
        self.dfs(graph, s, step)
        return self.visited

print("\n", Solution().reachable_in_n_steps(g, 0, 2))


# T(V, E) = O(|V| + |E|), S(V, E) = O(|V|), K = O(15)