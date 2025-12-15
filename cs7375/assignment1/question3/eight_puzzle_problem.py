import pandas as pd
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Utility Functions for Output
# ============================================================

def show_results(path, algorithm, display_type='Text'):
    print(f"\n{algorithm} Solution Path:")
    for depth, grid in enumerate(path):
        if display_type == 'Text':
            print(f"Depth: {depth}")
            print(grid.to_string(index=False, header=False))
            print("--------")
        else:
            plt.figure(figsize=(3, 3))
            sns.heatmap(grid, annot=True, cbar=False, square=True, linewidths=1, linecolor="black", 
                        cmap="Blues",fmt="d", annot_kws={"size": 16})
            plt.title(f"{algorithm} - Depth {depth}")
            plt.show()
    else:
        print("Valid display type options are 'Text' or 'Graphical'.")
# ============================================================
# Core Puzzle Mechanics
# ============================================================

puzzle_dimension = 3   

def find_blank(df):
    pos = df.stack().eq(0)              #Find 0 position in dataframe grid
    _y, _x = pos[pos].index[0]
    return _y, _x

def generate_successors(df):
    successors = []     # Generate dataframes for all valid move successor states
    x, y = find_blank(df)

    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down

    for d_x, d_y in moves:
        n_x, n_y = x + d_x, y + d_y
        if 0 <= n_x < puzzle_dimension and 0 <= n_y < puzzle_dimension:
            new_df = pd.DataFrame(df)
            new_df.iloc[x, y], new_df.iloc[n_x, n_y] = new_df.iloc[n_x, n_y], new_df.iloc[x, y]
            successors.append(new_df)

    return successors

def reconstruct_path(state, parents):  # use parent mapping to reconstruct path from initial state to goal state
    path = []
    while state is not None:
        path.append(state)
        parent = parents.get(state.to_json(), None)
        state = parent
    return list(reversed(path))

def solve_bfs(initial_state, goal_state):   #FIFO, does not use depth limit
    frontier = deque([initial_state])
    visited = {initial_state.to_json()}     #used the e
    parents = {initial_state.to_json(): None}

    while frontier:
        grid = frontier.popleft()
        if grid.equals(goal_state):
            return reconstruct_path(grid, parents)

        for successor in generate_successors(grid):
            key = successor.to_json()
            if key not in visited:
                visited.add(key)
                parents[key] = grid
                frontier.append(successor)
    return None

def solve_dfs(initial_state, goal_state, max_depth=20):  #LIFO, uses depth limit
    if max_depth > 50:
        max_depth = 50

    stack = [(initial_state, 0)]
    visited = {initial_state.to_json()}
    parents = {initial_state.to_json(): None}

    while stack:
        grid, depth = stack.pop()
        if grid.equals(goal_state):
            return reconstruct_path(grid, parents)

        if depth < max_depth:
            for successor in generate_successors(grid):
                key = successor.to_json()
                if key not in visited:
                    visited.add(key)
                    parents[key] = grid
                    stack.append((successor, depth + 1))
    return None

# ============================================================
# Driver
# ============================================================

if __name__ == "__main__":
    initial_state = pd.DataFrame([[2, 8, 3],[1, 6, 4],[7, 0, 5]])

    final_goal_state = pd.DataFrame([[1, 2, 3],[8, 0, 4], [7, 6, 5]])

    _bfs = solve_bfs(initial_state, final_goal_state)
    if _bfs:
        show_results(_bfs, "BFS","Graphical")

    _dfs = solve_dfs(initial_state, final_goal_state, max_depth=20)
    if _dfs:
        show_results(_dfs, "DFS", "Text")
