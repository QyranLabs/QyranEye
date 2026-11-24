from collections import deque

graph = {
    1: [2, 3],
    2: [1, 4, 5],
    3: [1, 6],
    4: [2],
    5: [2, 6],
    6: [3, 5]
}

def bfs(g, start):
    visited = [start]
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbor in g[vertex]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return visited

print("BFS from vertex 1:", bfs(graph, 1))
print("BFS from vertex 4:", bfs(graph, 4))