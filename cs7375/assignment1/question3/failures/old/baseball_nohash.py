import copy
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

class PuzzleState:
    """
    Represents a single state of the 8-puzzle using Python lists.
    """
    def __init__(self, grid, depth=0, parent=None):
        self.grid = grid
        self.depth = depth
        self.parent = parent

    def __eq__(self, other):
        return self.grid == other.grid

    def find_blank(self):
        """Locate the blank (0) position in the grid."""
        for i in range(3):
            for j in range(3):
                if self.grid[i][j] == 0:
                    return i, j

    def generate_successors(self):
        """Generate all possible moves from the current state."""
        successors = []
        x, y = self.find_blank()
        moves = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_grid = copy.deepcopy(self.grid)
                new_grid[x][y], new_grid[nx][ny] = new_grid[nx][ny], new_grid[x][y]
                successors.append(PuzzleState(new_grid, self.depth+1, self))
        return successors


class PuzzleSolver:
    """
    Solver for the 8-puzzle using BFS and DFS with list-based visited tracking.
    """
    def __init__(self, initial_state, final_goal_state):
        self.initial_state = PuzzleState(initial_state)
        self.goal_state = PuzzleState(final_goal_state)

    def is_goal(self, state):
        return state == self.goal_state

    def reconstruct_path(self, state):
        """Reconstruct path from initial to goal by following parents."""
        path = []
        while state:
            path.append(state)
            state = state.parent
        return list(reversed(path))

    def is_visited(self, state, visited):
        """Check if a state is already in the visited list."""
        return any(state == v for v in visited)

    def solve_bfs(self):
        """Solve puzzle using Breadth-First Search."""
        frontier = deque([self.initial_state])
        visited = [self.initial_state]

        while frontier:
            current = frontier.popleft()
            if self.is_goal(current):
                return self.reconstruct_path(current)

            for neighbor in current.generate_successors():
                if not self.is_visited(neighbor, visited):
                    visited.append(neighbor)
                    frontier.append(neighbor)
        return None

    def solve_dfs(self, max_depth=50):
        """Solve puzzle using Depth-First Search (with depth limit)."""
        stack = [self.initial_state]
        visited = [self.initial_state]

        while stack:
            current = stack.pop()
            if self.is_goal(current):
                return self.reconstruct_path(current)

            if current.depth < max_depth:
                for neighbor in current.generate_successors():
                    if not self.is_visited(neighbor, visited):
                        visited.append(neighbor)
                        stack.append(neighbor)
        return None

    # ---------- Output Functions ----------

    def print_path_text(self, path, algo_name):
        """Prints the solution path in text format with depth info."""
        print(f"\n{algo_name} Solution Path:")
        for state in path:
            print(f"Depth: {state.depth}")
            for row in state.grid:
                print(" ".join(map(str, row)))
            print("--------")

    def plot_path_seaborn(self, path, algo_name):
        """Plots each state in the path using seaborn heatmaps."""
        for state in path:
            plt.figure(figsize=(3,3))
            sns.heatmap(state.grid, annot=True, cbar=False, square=True,
                        linewidths=1, linecolor="black", cmap="Blues",
                        fmt="d", annot_kws={"size":16})
            plt.title(f"{algo_name} - Depth {state.depth}")
            plt.show()


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    initial_state = [[2, 8, 3],
                     [1, 6, 4],
                     [7, 0, 5]]

    final_goal_state = [[1, 2, 3],
                        [8, 0, 4],
                        [7, 6, 5]]

    solver = PuzzleSolver(initial_state, final_goal_state)

    # Solve with BFS
    bfs_path = solver.solve_bfs()
    if bfs_path:
        solver.print_path_text(bfs_path, "BFS")
        solver.plot_path_seaborn(bfs_path, "BFS")

    # Solve with DFS
    dfs_path = solver.solve_dfs(max_depth=30)
    if dfs_path:
        solver.print_path_text(dfs_path, "DFS")
        solver.plot_path_seaborn(dfs_path, "DFS")
