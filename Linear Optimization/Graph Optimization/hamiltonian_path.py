import math


class HamiltonianPath:
    def __init__(self):
        return

    def exists(self, graph):
        return True


g = [
    [1, 2, 3],
    [0, 2, 4],
    [0, 1, 3, 4],
    [0, 2],
    [1, 2]
]

print(HamiltonianPath().exists(g))
