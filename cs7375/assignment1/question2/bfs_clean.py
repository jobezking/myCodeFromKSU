from collections import deque

graph = {
    1: {"neighbors": [2, 6], "node_weight": 11},
    2: {"neighbors": [1, 6, 3], "node_weight": 3},
    3: {"neighbors": [2, 8, 7], "node_weight": 1},
    4: {"neighbors": [8, 7, 9], "node_weight": 9},
    5: {"neighbors": [7, 9], "node_weight": 10},
    6: {"neighbors": [8, 1, 2, 7], "node_weight": 5},
    7: {"neighbors": [3, 6, 4, 5], "node_weight": 2},
    8: {"neighbors": [3, 4, 6], "node_weight": 1},
    9: {"neighbors": [4, 5], "node_weight": 7}
}

def bfs_weighted(graph, start):
    print(f"--- Starting BFS from node {start} ---")
    
    visited = {node: False for node in graph}
    queue = deque([start])
    visited[start] = True

    total_weight = graph[start]["node_weight"]
    print(f"Initial weight from start node {start}: {total_weight}\n")

    while queue:
        u = queue.popleft()
        print(f"Processing node: {u}")

        for v in graph[u]["neighbors"]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                
                node_weight_v = graph[v]["node_weight"]
                total_weight += node_weight_v
                
                print(f"  -> Found unvisited neighbor: {v}. Adding its weight ({node_weight_v}).")
                print(f"     New total weight: {total_weight}")
            else:
                print(f"  -> Neighbor {v} already visited. Skipping.")
        print("-" * 20)

    print("--- BFS Complete ---")
    print(f"Final Total Weight: {total_weight}")

if __name__ == "__main__":
    bfs_weighted(graph, 8)