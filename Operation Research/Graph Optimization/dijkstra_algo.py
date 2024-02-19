import heapq as heap
import math


def dijkstra(graph, s, e=None):
    visited = set()
    parents_map = dict()

    dist = [math.inf for _ in range(len(graph))]
    dist[s] = 0

    priority_queue = []
    heap.heappush(priority_queue, (0, s))  # min heap

    while len(priority_queue) > 0:
        cur_weight, cur = heap.heappop(priority_queue)
        if cur in visited:  # non ha senso rivisitare --> la prima volta che arrivi, il costo è ottimale
            continue
        visited.add(cur)

        if cur == e:  # nel caso che tu abbia un solo nodo di riferimento ti fermi quando lo visiti --> non lo cambierai mai più
            return parents_map, dist

        for node, weight in graph[cur]:
            new_cost = cur_weight + weight
            heap.heappush(priority_queue, (new_cost, node))  # lo aggiungi anche più volte, ma tanto lo visiti solo una volta
            if new_cost < dist[node]:
                parents_map[node] = cur  # in questo modo, alla fine potrò tornare indietro e trovare il path effettivo
                dist[node] = new_cost

    return parents_map, dist

# T(N) = O(|V|^2), S(N) = O(|V|)
