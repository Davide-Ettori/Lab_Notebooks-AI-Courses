import math

def print_mat(mat):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            print(mat[i][j], end=" ")
        print()
def floyd_warshall(graph):
    dist = list()
    pred = list()
    for i in range(len(graph)):
        dist.append(list())
        pred.append(list())
        for j in range(len(graph[i])):
            dist[-1].append(math.inf)
            pred[-1].append(i)

    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if i == j:
                dist[i][j] = 0
                continue
            if math.isinf(graph[i][j]):
                continue
            dist[i][j] = graph[i][j]

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                if i == k or j == k:
                    continue

                if dist[i][k] + dist[k][j] < dist[i][j]:  # classica condizione di rilassamento
                    dist[i][j] = dist[i][k] + dist[k][j]  # update chiaro, tipico del DP
                    pred[i][j] = pred[k][j]  # update chiaro passaggio triangolare

    for i in range(len(graph)):
        if dist[i][i] < 0:
            return None, None  # cicli negativi, problema mal posto

    return dist, pred

g = [
    [math.inf, 4, 3],
    [2, math.inf, 5],
    [-2, -4, math.inf]
]

dist, pred = floyd_warshall(g)

for i in range(len(dist)):
    for j in range(len(dist[i])):
        pred[i][j] += 1  # aggiusto i risultati con il grafo che sto considerando

print()
print_mat(dist)
print()
print_mat(pred)