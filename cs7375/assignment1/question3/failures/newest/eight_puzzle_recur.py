import copy
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Core Puzzle Functions ----------

def find_blank(grid):
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                return i, j

def generate_successors(grid):
    successors = []
    x, y = find_blank(grid)
    moves = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_grid = [r[:] for r in grid]
            new_grid[x][y], new_grid[nx][ny] = new_grid[nx][ny], new_grid[x][y]
            successors.append(new_grid)
    return successors

def is_goal(grid, goal_state):
    return grid == goal_state

def reconstruct_path(state, parents):
    path = []
    # Convert state to tuple for dictionary lookup if it's not already
    current_key = tuple(map(tuple, state))
    while current_key is not None:
        # Convert tuple back to list of lists for the path
        path.append(list(map(list, current_key)))
        current_state_list = list(map(list, current_key))
        # Get the parent state (which is a list of lists)
        parent_state_list = parents.get(current_key)
        if parent_state_list is not None:
            # Convert parent to tuple for the next lookup
            current_key = tuple(map(tuple, parent_state_list))
        else:
            current_key = None
    return list(reversed(path))


# ---------- BFS (Recursive) ----------

def solve_bfs_recursive(initial_state, goal_state):
    """Wrapper function to initialize and start the recursive BFS."""
    frontier = [initial_state]  # Start with a "level" containing only the initial state
    visited = {tuple(map(tuple, initial_state))}
    parents = {tuple(map(tuple, initial_state)): None}

    goal_found = _bfs_helper(frontier, goal_state, visited, parents)

    if goal_found:
        return reconstruct_path(goal_found, parents)
    return None

def _bfs_helper(current_level_nodes, goal_state, visited, parents):
    """The recursive helper that processes the search one level at a time."""
    if not current_level_nodes:
        return None

    next_level_nodes = []
    for grid in current_level_nodes:
        if is_goal(grid, goal_state):
            return grid

        for successor in generate_successors(grid):
            key = tuple(map(tuple, successor))
            if key not in visited:
                visited.add(key)
                parents[key] = grid
                next_level_nodes.append(successor)

    return _bfs_helper(next_level_nodes, goal_state, visited, parents)

# ---------- DFS (Recursive) ----------

def solve_dfs_recursive(initial_state, goal_state, max_depth=50):
    """Wrapper function to initialize and start the recursive DFS."""
    visited = {tuple(map(tuple, initial_state))}
    parents = {tuple(map(tuple, initial_state)): None}
    
    goal_found = _dfs_helper(initial_state, goal_state, visited, parents, 0, max_depth)
    
    if goal_found:
        return reconstruct_path(goal_found, parents)
    return None

def _dfs_helper(current_grid, goal_state, visited, parents, depth, max_depth):
    """The recursive helper function that performs the DFS."""
    if is_goal(current_grid, goal_state):
        return current_grid

    if depth >= max_depth:
        return None

    for successor in generate_successors(current_grid):
        key = tuple(map(tuple, successor))
        if key not in visited:
            visited.add(key)
            parents[key] = current_grid
            
            result = _dfs_helper(successor, goal_state, visited, parents, depth + 1, max_depth)
            
            if result is not None:
                return result
    
    return None

# ---------- Output Functions ----------

def print_path_text(path, algo_name):
    print(f"\n{algo_name} Solution Path:")
    if not path:
        print("No solution found.")
        return
    for depth, grid in enumerate(path):
        print(f"Depth: {depth}")
        for row in grid:
            print(" ".join(map(str, row)))
        print("--------")

def plot_path_seaborn(path, algo_name):
    if not path:
        return
    for depth, grid in enumerate(path):
        plt.figure(figsize=(3,3))
        sns.heatmap(grid, annot=True, cbar=False, square=True,
                    linewidths=1, linecolor="black", cmap="Blues",
                    fmt="d", annot_kws={"size":16})
        plt.title(f"{algo_name} - Depth {depth}")
        plt.show()

# ---------- Driver ----------

if __name__ == "__main__":
    initial_state = [[2, 8, 3],
                     [1, 6, 4],
                     [7, 0, 5]]

    final_goal_state = [[1, 2, 3],
                        [8, 0, 4],
                        [7, 6, 5]]

    # Recursive BFS
    print("--- Running Recursive BFS ---")
    bfs_path_rec = solve_bfs_recursive(initial_state, final_goal_state)
    if bfs_path_rec:
        print(f"BFS found a solution in {len(bfs_path_rec)-1} steps.")
        print_path_text(bfs_path_rec, "Recursive BFS")
        # plot_path_seaborn(bfs_path_rec, "Recursive BFS") # Uncomment to see plots

    # Recursive DFS
    print("\n--- Running Recursive DFS ---")
    # Note: A simple recursive DFS may not find the shortest path and can be slow.
    # A depth limit is crucial.
    dfs_path_rec = solve_dfs_recursive(initial_state, final_goal_state, max_depth=30)
    if dfs_path_rec:
        print(f"DFS found a solution in {len(dfs_path_rec)-1} steps.")
        print_path_text(dfs_path_rec, "Recursive DFS")
        # plot_path_seaborn(dfs_path_rec, "Recursive DFS") # Uncomment to see plots