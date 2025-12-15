# bfs.py

# This script demonstrates a modified Breadth-First Search (BFS) algorithm.
# BFS is a graph traversal algorithm that explores nodes "layer by layer."
# This specific implementation not only traverses the graph but also calculates
# the sum of the weights of all visited nodes.

# We import the 'deque' object from the 'collections' library. A deque
# (pronounced "deck") is a double-ended queue. We use it for our BFS queue
# because it's highly efficient for adding items to the end (append) and
# removing them from the front (popleft), which is exactly the "First-In,
# First-Out" (FIFO) behavior that BFS requires.
from collections import deque

# This is our graph, represented as an adjacency list using a Python dictionary.
# - The keys (1, 2, 3, etc.) are the nodes (or vertices) of the graph.
# - The value for each key is another dictionary with two pieces of info:
#   - "neighbors": A list of all nodes directly connected to this node.
#   - "node_weight": A numerical value associated with the node itself.
# Think of it like a map of cities: each number is a city, "neighbors" are
# direct roads to other cities, and "node_weight" is the city's population.
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
    """
    Performs a Breadth-First Search on a graph starting from a given node
    and calculates the sum of the weights of all visited nodes.
    Includes verbose output for demonstration purposes.
    """
    print(f"--- Starting BFS from node {start} ---")
    
    # Initialization
    visited = {node: False for node in graph}
    queue = deque([start])
    visited[start] = True

    # Initialize total_weight with the weight of the starting node
    total_weight = graph[start]["node_weight"]
    print(f"Initial weight from start node {start}: {total_weight}\n")

    while queue:
        # Dequeue the node at the front of the queue
        u = queue.popleft()
        print(f"Processing node: {u}")

        # Explore its neighbors
        for v in graph[u]["neighbors"]:
            # Process the neighbor 'v' only if it has NOT been visited before
            if not visited[v]:
                # Mark as visited, add to queue, and update total weight
                visited[v] = True
                queue.append(v)
                
                # Add the weight of the newly discovered node
                node_weight_v = graph[v]["node_weight"]
                total_weight += node_weight_v
                
                # **NEW**: Add meaningful print statements for this step
                print(f"  -> Found unvisited neighbor: {v}. Adding its weight ({node_weight_v}).")
                print(f"     New total weight: {total_weight}")
            else:
                # **NEW**: Indicate that a neighbor has already been seen
                print(f"  -> Neighbor {v} already visited. Skipping.")
        print("-" * 20) # Separator for clarity

    print("--- BFS Complete ---")
    print(f"Final Total Weight: {total_weight}")

# --- Example Usage ---
if __name__ == "__main__":
    # The new sample run starts from node 8 as requested.
    bfs_weighted(graph, 8)
