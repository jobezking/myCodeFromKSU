import pandas as pd
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Utility Functions for Output
# ============================================================

def print_path_text(path, algo_name):
    """
    Print the solution path in text form, showing each depth level.
    """
    print(f"\n{algo_name} Solution Path:")
    for depth, df in enumerate(path):
        print(f"Depth: {depth}")
        print(df.to_string(header=False, index=False))
        print("--------------")

def plot_path_seaborn(path, algo_name):
    """
    Visualize the solution path using seaborn heatmaps.
    Each state is shown as a 3x3 grid with annotations. [cite: 2]
    """
    for depth, df in enumerate(path):
        plt.figure(figsize=(3, 3))
        sns.heatmap(df, annot=True, cbar=False, square=True,
                    linewidths=1, linecolor="black", cmap="Blues",
                    fmt="d", annot_kws={"size": 16})
        plt.title(f"{algo_name} - Depth {depth}")
        plt.show()

# ============================================================
# Core Puzzle Mechanics
# ============================================================

N = 3  # Puzzle dimension (3x3 for 8-puzzle)

def find_blank(df):
    """Locate the blank (0) position in the DataFrame."""
    for i in range(N):
        for j in range(N):
            if df.iat[i, j] == 0:
                return i, j

def generate_successors(df):
    """
    Generate all valid successor states by sliding the blank
    (0) up, down, left, or right. [cite: 4]
    """
    successors = []
    x, y = find_blank(df)

    # Direction vectors: Left, Right, Up, Down
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < N and 0 <= ny < N:
            # Copy DataFrame and swap blank with neighbor
            new_df = df.copy()
            new_df.iat[x, y], new_df.iat[nx, ny] = new_df.iat[nx, ny], new_df.iat[x, y]
            successors.append(new_df)

    return successors

def reconstruct_path(state_df, parents):
    """
    Reconstruct the path from the initial state to the goal
    using the parent dictionary. [cite: 7]
    """
    path = []
    # Use a hashable representation (tuple of tuples) of the DataFrame as a key
    current_key = tuple(map(tuple, state_df.values))
    while current_key is not None:
        # Retrieve the DataFrame and its parent's key
        df, parent_key = parents[current_key]
        path.append(df)
        current_key = parent_key
    return list(reversed(path))

# Helper function to check if a DataFrame is in a list of DataFrames
def is_state_in_list(state_df, df_list):
    """Check for DataFrame existence in a list using .equals()."""
    for item in df_list:
        if item.equals(state_df):
            return True
    return False

# ============================================================
# Breadth-First Search (BFS)
# ============================================================

def solve_bfs(initial_state, goal_state):
    """
    Solve the 8-puzzle using Breadth-First Search (BFS).
    BFS explores the shallowest nodes first using a FIFO queue. [cite: 9]
    """
    frontier = deque([initial_state])  # FIFO queue [cite: 10]
    # Use a dictionary for O(1) average time complexity lookups for visited states
    # Key: hashable tuple representation, Value: the DataFrame itself
    visited = {tuple(map(tuple, initial_state.values)): initial_state}
    # Parent dictionary for efficient path reconstruction
    # Key: child state tuple, Value: (child_df, parent_state_tuple)
    parents = {tuple(map(tuple, initial_state.values)): (initial_state, None)}

    while frontier:
        grid_df = frontier.popleft()
        if grid_df.equals(goal_state):
            return reconstruct_path(grid_df, parents)

        for successor_df in generate_successors(grid_df):
            successor_key = tuple(map(tuple, successor_df.values))
            if successor_key not in visited:
                visited[successor_key] = successor_df
                grid_key = tuple(map(tuple, grid_df.values))
                parents[successor_key] = (successor_df, grid_key)
                frontier.append(successor_df)

    return None  # No solution found

# ============================================================
# Depth-First Search (DFS) with Depth Limit
# ============================================================

def solve_dfs(initial_state, goal_state, max_depth=20):
    """
    Solve the 8-puzzle using Depth-First Search (DFS).
    DFS explores deeply before backtracking, using a LIFO stack. [cite: 13]
    A depth limit prevents infinite loops. [cite: 14]
    """
    if max_depth > 50:
        max_depth = 50
    stack = [(initial_state, 0)]  # (state_df, depth)
    visited = {tuple(map(tuple, initial_state.values)): initial_state}
    parents = {tuple(map(tuple, initial_state.values)): (initial_state, None)}

    while stack:
        grid_df, depth = stack.pop()
        if grid_df.equals(goal_state):
            return reconstruct_path(grid_df, parents)

        if depth < max_depth:
            for successor_df in generate_successors(grid_df):
                successor_key = tuple(map(tuple, successor_df.values))
                if successor_key not in visited:
                    visited[successor_key] = successor_df
                    grid_key = tuple(map(tuple, grid_df.values))
                    parents[successor_key] = (successor_df, grid_key)
                    stack.append((successor_df, depth + 1))

    return None  # No solution found within depth limit

# ============================================================
# Driver
# ============================================================

if __name__ == "__main__":
    # Example initial and goal states as pandas DataFrames
    initial_data = [[2, 8, 3],
                    [1, 6, 4],
                    [7, 0, 5]]
    initial_state_df = pd.DataFrame(initial_data)

    goal_data = [[1, 2, 3],
                 [8, 0, 4],
                 [7, 6, 5]]
    final_goal_state_df = pd.DataFrame(goal_data)

    print("Initial State:")
    print(initial_state_df.to_string(header=False, index=False))
    
    print("\nGoal State:")
    print(final_goal_state_df.to_string(header=False, index=False))

    # Run BFS
    bfs_path = solve_bfs(initial_state_df, final_goal_state_df)
    if bfs_path:
        print_path_text(bfs_path, "BFS")
        # plot_path_seaborn(bfs_path, "BFS") # Uncomment to display plots

    # Run DFS with depth limit
    dfs_path = solve_dfs(initial_state_df, final_goal_state_df, max_depth=20)
    if dfs_path:
        print_path_text(dfs_path, "DFS")
        # plot_path_seaborn(dfs_path, "DFS") # Uncomment to display plots