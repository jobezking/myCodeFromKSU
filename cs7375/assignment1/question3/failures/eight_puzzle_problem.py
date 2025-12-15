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
    for depth, grid in enumerate(path):
        print(f"Depth: {depth}")
        for row in grid:
            print(" ".join(map(str, row)))
        print("--------")

def plot_path_seaborn(path, algo_name):
    """
    Visualize the solution path using seaborn heatmaps.
    Each state is shown as a 3x3 grid with annotations.
    """
    for depth, grid in enumerate(path):
        plt.figure(figsize=(3, 3))
        sns.heatmap(grid, annot=True, cbar=False, square=True,
                    linewidths=1, linecolor="black", cmap="Blues",
                    fmt="d", annot_kws={"size": 16})
        plt.title(f"{algo_name} - Depth {depth}")
        plt.show()

# ============================================================
# Core Puzzle Mechanics
# ============================================================

N = 3  # Puzzle dimension (3x3 for 8-puzzle)

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

def reconstruct_path(state, parents):
    """
    Reconstruct the path from the initial state to the goal
    using the parent list.
    """
    path = []
    while state is not None:
        path.append(state)
        # Find parent in the list of (child, parent) pairs
        parent = None
        for child, par in parents:
            if child == state:
                parent = par
                break
        state = parent
    return list(reversed(path))

# ============================================================
# Breadth-First Search (BFS)
# ============================================================

def solve_bfs(initial_state, goal_state):
    """
    Solve the 8-puzzle using Breadth-First Search (BFS).
    BFS explores the shallowest nodes first using a FIFO queue.
    """
    frontier = deque([initial_state])  # FIFO queue
    visited = [initial_state]          # List of visited states
    parents = [(initial_state, None)]  # List of (state, parent)

    while frontier:
        grid = frontier.popleft()  # Dequeue from front
        if grid == goal_state:
            return reconstruct_path(grid, parents)

        for successor in generate_successors(grid):
            if successor not in visited:  # Linear-time membership check
                visited.append(successor)
                parents.append((successor, grid))
                frontier.append(successor)  # Enqueue at back

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
    """
    if max_depth > 50:
       max_depth = 50
    stack = [(initial_state, 0)]       # (state, depth)
    visited = [initial_state]          # List of visited states
    parents = [(initial_state, None)]  # List of (state, parent)

    while stack:
        grid, depth = stack.pop()  # LIFO stack pop
        if grid == goal_state:
            return reconstruct_path(grid, parents)

        if depth < max_depth:  # Boundary check
            for successor in generate_successors(grid):
                if successor not in visited:  # Linear-time membership check
                    visited.append(successor)
                    parents.append((successor, grid))
                    stack.append((successor, depth + 1))

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
    bfs_path = solve_bfs(initial_state, final_goal_state)
    if bfs_path:
        print_path_text(bfs_path, "BFS")
        plot_path_seaborn(bfs_path, "BFS")

    # Run DFS with depth limit
    dfs_path = solve_dfs(initial_state, final_goal_state, max_depth=20)
    if dfs_path:
        print_path_text(dfs_path, "DFS")
        plot_path_seaborn(dfs_path, "DFS")
