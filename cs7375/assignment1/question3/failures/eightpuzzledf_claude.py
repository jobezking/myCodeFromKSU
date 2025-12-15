import pandas as pd
import numpy as np
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Utility Functions for Output
# ============================================================

def print_path_text(path_df, algo_name):
    """
    Print the solution path in text form, showing each depth level.
    path_df: DataFrame with columns ['depth', 'state']
    """
    print(f"\n{algo_name} Solution Path:")
    for idx, row in path_df.iterrows():
        print(f"Depth: {row['depth']}")
        state = row['state']
        for grid_row in state:
            print(" ".join(map(str, grid_row)))
        print("--------")

def plot_path_seaborn(path_df, algo_name):
    """
    Visualize the solution path using seaborn heatmaps.
    Each state is shown as a 3x3 grid with annotations.
    """
    for idx, row in path_df.iterrows():
        plt.figure(figsize=(3, 3))
        grid = np.array(row['state'])
        sns.heatmap(grid, annot=True, cbar=False, square=True,
                    linewidths=1, linecolor="black", cmap="Blues",
                    fmt="d", annot_kws={"size": 16})
        plt.title(f"{algo_name} - Depth {row['depth']}")
        plt.show()

# ============================================================
# Core Puzzle Mechanics
# ============================================================

N = 3  # Puzzle dimension (3x3 for 8-puzzle)

def state_to_tuple(grid):
    """Convert grid to hashable tuple for efficient lookups."""
    return tuple(tuple(row) for row in grid)

def tuple_to_state(tup):
    """Convert tuple back to list grid."""
    return [list(row) for row in tup]

def find_blank(grid):
    """Locate the blank (0) position in the grid."""
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                return i, j

def generate_successors(grid):
    """
    Generate all valid successor states by sliding the blank
    (0) up, down, left, or right.
    """
    successors = []
    x, y = find_blank(grid)

    # Direction vectors: Left, Right, Up, Down
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < N and 0 <= ny < N:
            # Copy grid and swap blank with neighbor
            new_grid = [row[:] for row in grid]
            new_grid[x][y], new_grid[nx][ny] = new_grid[nx][ny], new_grid[x][y]
            successors.append(new_grid)

    return successors

def reconstruct_path(state, parents_df):
    """
    Reconstruct the path from the initial state to the goal
    using the parents DataFrame.
    parents_df: DataFrame with columns ['state_tuple', 'parent_tuple', 'depth']
    """
    path_data = []
    current_tuple = state_to_tuple(state)
    
    while current_tuple is not None:
        # Find the row for current state
        row = parents_df[parents_df['state_tuple'] == current_tuple]
        if not row.empty:
            depth = row.iloc[0]['depth']
            path_data.append({
                'depth': depth,
                'state': tuple_to_state(current_tuple)
            })
            current_tuple = row.iloc[0]['parent_tuple']
        else:
            break
    
    # Reverse to get path from initial to goal
    path_df = pd.DataFrame(path_data[::-1])
    return path_df

# ============================================================
# Breadth-First Search (BFS)
# ============================================================

def solve_bfs(initial_state, goal_state):
    """
    Solve the 8-puzzle using Breadth-First Search (BFS).
    BFS explores the shallowest nodes first using a FIFO queue.
    
    Returns: DataFrame with columns ['depth', 'state']
    """
    initial_tuple = state_to_tuple(initial_state)
    goal_tuple = state_to_tuple(goal_state)
    
    # Frontier queue stores (state_tuple, depth)
    frontier = deque([(initial_tuple, 0)])
    
    # Visited set for O(1) lookups
    visited = {initial_tuple}
    
    # Parents DataFrame to track the search tree
    parents_data = [{
        'state_tuple': initial_tuple,
        'parent_tuple': None,
        'depth': 0
    }]
    parents_df = pd.DataFrame(parents_data)

    while frontier:
        current_tuple, depth = frontier.popleft()  # Dequeue from front
        
        if current_tuple == goal_tuple:
            return reconstruct_path(tuple_to_state(current_tuple), parents_df)

        current_state = tuple_to_state(current_tuple)
        for successor in generate_successors(current_state):
            successor_tuple = state_to_tuple(successor)
            
            if successor_tuple not in visited:
                visited.add(successor_tuple)
                
                # Add to parents DataFrame
                new_row = pd.DataFrame([{
                    'state_tuple': successor_tuple,
                    'parent_tuple': current_tuple,
                    'depth': depth + 1
                }])
                parents_df = pd.concat([parents_df, new_row], ignore_index=True)
                
                frontier.append((successor_tuple, depth + 1))

    return None  # No solution found

# ============================================================
# Depth-First Search (DFS) with Depth Limit
# ============================================================

def solve_dfs(initial_state, goal_state, max_depth=20):
    """
    Solve the 8-puzzle using Depth-First Search (DFS).
    DFS explores as deep as possible along each branch
    before backtracking, using a LIFO stack.

    A maximum depth is enforced to prevent infinite descent.
    
    Returns: DataFrame with columns ['depth', 'state']
    """
    if max_depth > 50:
        max_depth = 50
    
    initial_tuple = state_to_tuple(initial_state)
    goal_tuple = state_to_tuple(goal_state)
    
    # Stack stores (state_tuple, depth)
    stack = [(initial_tuple, 0)]
    
    # Visited set for O(1) lookups
    visited = {initial_tuple}
    
    # Parents DataFrame to track the search tree
    parents_data = [{
        'state_tuple': initial_tuple,
        'parent_tuple': None,
        'depth': 0
    }]
    parents_df = pd.DataFrame(parents_data)

    while stack:
        current_tuple, depth = stack.pop()  # LIFO stack pop
        
        if current_tuple == goal_tuple:
            return reconstruct_path(tuple_to_state(current_tuple), parents_df)

        if depth < max_depth:  # Boundary check
            current_state = tuple_to_state(current_tuple)
            for successor in generate_successors(current_state):
                successor_tuple = state_to_tuple(successor)
                
                if successor_tuple not in visited:
                    visited.add(successor_tuple)
                    
                    # Add to parents DataFrame
                    new_row = pd.DataFrame([{
                        'state_tuple': successor_tuple,
                        'parent_tuple': current_tuple,
                        'depth': depth + 1
                    }])
                    parents_df = pd.concat([parents_df, new_row], ignore_index=True)
                    
                    stack.append((successor_tuple, depth + 1))

    return None  # No solution found within depth limit

# ============================================================
# Driver
# ============================================================

if __name__ == "__main__":
    # Example initial and goal states
    initial_state = [[2, 8, 3],
                     [1, 6, 4],
                     [7, 0, 5]]

    final_goal_state = [[1, 2, 3],
                        [8, 0, 4],
                        [7, 6, 5]]

    # Run BFS
    print("Running BFS...")
    bfs_path = solve_bfs(initial_state, final_goal_state)
    if bfs_path is not None:
        print(f"\nBFS found solution with {len(bfs_path)} steps")
        print("\nBFS Path DataFrame:")
        print(bfs_path)
        print_path_text(bfs_path, "BFS")
        plot_path_seaborn(bfs_path, "BFS")
    else:
        print("BFS: No solution found")

    # Run DFS with depth limit
    print("\nRunning DFS...")
    dfs_path = solve_dfs(initial_state, final_goal_state, max_depth=20)
    if dfs_path is not None:
        print(f"\nDFS found solution with {len(dfs_path)} steps")
        print("\nDFS Path DataFrame:")
        print(dfs_path)
        print_path_text(dfs_path, "DFS")
        plot_path_seaborn(dfs_path, "DFS")
    else:
        print("DFS: No solution found within depth limit")