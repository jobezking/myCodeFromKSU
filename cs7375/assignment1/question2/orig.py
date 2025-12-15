graph = {
    1: [2, 6],
    2: [1, 6, 3], 
    3: [2, 8, 7], 
    4: [8, 7, 9], 
    5: [7, 9], 
    6: [8, 1, 2, 7], 
    7:  [3, 6, 4, 5],
    8:  [3, 4, 6],
    9: [4, 5]
}

visited = {node: False for node in graph}

def dfs(at_node):         # recursive DFS function
    if(visited[at_node]):
        return
    visited[at_node] = True
    print(f"Visiting {at_node}")

    neighbors = graph[at_node]
    for nxt in neighbors:
        dfs(nxt)