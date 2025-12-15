# Depth First Search (DFS) implementation in Python
# The graph is represented as an adjacency list using a dictionary
# Each key is a node, and its value is a list of neighboring nodes
graph = {
    1: {"neighbors": [2, 6], "weight": 11},
    2: {"neighbors": [1, 6, 3], "weight": 3},
    3: {"neighbors": [2, 8, 7], "weight": 1},
    4: {"neighbors": [8, 7, 9], "weight": 9},
    5: {"neighbors": [7, 9], "weight": 10},
    6: {"neighbors": [8, 1, 2, 7], "weight": 5},
    7: {"neighbors": [3, 6, 4, 5], "weight": 2},
    8: {"neighbors": [3, 4, 6], "weight": 1},
    9: {"neighbors": [4, 5], "weight": 7}
}
visited = {node: False for node in graph}

def dfs(at_node, node_sum=0):         # recursive DFS function
    if(visited[at_node]):
        return node_sum
    visited[at_node] = True
    node_sum += graph[at_node]["weight"]
    print(f"Visiting {at_node}, weight={graph[at_node]['weight']}, running sum={node_sum}")

    neighbors = graph[at_node]["neighbors"]
    for next in neighbors:
        if not visited[next]:
            print(f" {at_node} -> {next} (going deeper)")
        node_sum = dfs(next, node_sum) 
    else:
        print(f" {at_node} -> {next} (already visited)")
    
    return node_sum
###
final_node_sum = dfs(8)
print("\nFinal total weight of visited nodes:", final_node_sum)