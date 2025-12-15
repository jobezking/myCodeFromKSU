import copy
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Core Puzzle Functions ----------

N = 3
# Direction vectors: Left, Right, Up, Down
row = [0, 0, -1, 1]
col = [-1, 1, 0, 0]

def find_blank(grid):
    """Locate the blank (0) position in the grid."""
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                return i, j

def is_goal(grid, goal_state):
    return grid == goal_state

def generate_successors(grid):
    """Generate all possible moves from the current state."""
    successors = []
    x, y = find_blank(grid)
    for i in range(4):
        nx, ny = x + row[i], y + col[i]
        if 0 <= nx < N and 0 <= ny < N:
            new_grid = [r[:] for r in grid]
            new_grid[x][y], new_grid[nx][ny] = new_grid[nx][ny], new_grid[x][y]
            successors.append(new_grid)
    return successors

def reconstruct_path(state, parents):
    """Reconstruct path from initial to goal using parent dictionary."""
    path = []
    while state is not None:
        path.append(state)
        state = parents.get(tuple(map(tuple, state)))
    return list(reversed(path))

# ---------- BFS and DFS ----------

def solve_bfs(initial_state, goal_state):
    """Breadth-first search using a queue."""
    q = deque()
    q.append(initial_state)

    visited = {tuple(map(tuple, initial_state)): True}
    parents = {tuple(map(tuple, initial_state)): None}

    while q:
        node = q.popleft()
        if is_goal(node, goal_state):
            return reconstruct_path(node, parents)

        for successor in generate_successors(node):
            key = tuple(map(tuple, successor))
            if key not in visited:
                visited[key] = True
                parents[key] = node
                q.append(successor)
    return None

def solve_dfs(initial_state, goal_state, max_depth=20):
    """Recursive depth-limited DFS using direction vectors."""

    visited = {tuple(map(tuple, initial_state)): False}
    parents = {tuple(map(tuple, initial_state)): None}
    solution_path = []

    def dfs(node, depth):
        nonlocal solution_path
        key = tuple(map(tuple, node))

        if visited.get(key, False):
            return False
        visited[key] = True

        if is_goal(node, goal_state):
            # reconstruct path manually
            path = []
            while node is not None:
                path.append(node)
                node = parents.get(tuple(map(tuple, node)))
            solution_path = list(reversed(path))
            return True

        if depth >= max_depth:
            return False

        x, y = find_blank(node)
        for i in range(4):
            nx, ny = x + row[i], y + col[i]
            if 0 <= nx < N and 0 <= ny < N:
                new_grid = [r[:] for r in node]
                new_grid[x][y], new_grid[nx][ny] = new_grid[nx][ny], new_grid[x][y]
                new_key = tuple(map(tuple, new_grid))
                if not visited.get(new_key, False):
                    parents[new_key] = node
                    if dfs(new_grid, depth + 1):
                        return True
        return False

    dfs(initial_state, 0)
    return solution_path if solution_path else None

# ---------- Output Functions ----------

def print_path_text(path, algo_name):
    print(f"\n{algo_name} Solution Path:")
    for depth, grid in enumerate(path):
        print(f"Depth: {depth}")
        for row in grid:
            print(" ".join(map(str, row)))
        print("--------")

def plot_path_seaborn(path, algo_name):
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

    # BFS
    bfs_path = solve_bfs(initial_state, final_goal_state)
    if bfs_path:
        print_path_text(bfs_path, "BFS")
        plot_path_seaborn(bfs_path, "BFS")

    # DFS with depth limit
    dfs_path = solve_dfs(initial_state, final_goal_state, max_depth=20)
    if dfs_path:
        print_path_text(dfs_path, "DFS")
        plot_path_seaborn(dfs_path, "DFS")
