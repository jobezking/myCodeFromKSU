import copy
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

class PuzzleState:
    """
    Represents a single state of the 8-puzzle.
    Stores the grid, depth, and parent for path reconstruction.
    """
    def __init__(self, grid, depth=0, parent=None):
        self.grid = grid  # 2D list representation
        self.depth = depth
        self.parent = parent

    def __eq__(self, other):
        return self.grid == other.grid

    def __hash__(self):
        # Convert grid to tuple of tuples for hashing
        return hash(tuple(tuple(row) for row in self.grid))

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
                # Swap blank with neighbor
                new_grid[x][y], new_grid[nx][ny] = new_grid[nx][ny], new_grid[x][y]
                successors.append(PuzzleState(new_grid, self.depth+1, self))
        return successors


class PuzzleSolver:
    """
    Solver for the 8-puzzle using BFS and DFS.
    """
    def __init__(self, initial_state_dict, goal_state_dict):
        self.initial_state = PuzzleState([initial_state_dict[i] for i in range(3)])
        self.goal_state = PuzzleState([goal_state_dict[i] for i in range(3)])

    def is_goal(self, state):
        return state == self.goal_state

    def reconstruct_path(self, state):
        """Reconstruct path from initial to goal by following parents."""
        path = []
        while state:
            path.append(state)
            state = state.parent
        return list(reversed(path))

    def solve_bfs(self):
        """Solve puzzle using Breadth-First Search."""
        frontier = deque([self.initial_state])
        visited = set([self.initial_state])

        while frontier:
            current = frontier.popleft()
            if self.is_goal(current):
                return self.reconstruct_path(current)

            for neighbor in current.generate_successors():
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return None

    def solve_dfs(self, max_depth=50):
        """Solve puzzle using Depth-First Search (with depth limit)."""
        stack = [self.initial_state]
        visited = set([self.initial_state])

        while stack:
            current = stack.pop()
            if self.is_goal(current):
                return self.reconstruct_path(current)

            if current.depth < max_depth:
                for neighbor in current.generate_successors():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        return None

    # ---------- Output Functions ----------

    def print_path_text(self, path):
        """Prints the solution path in text format with depth info."""
        print("Initial State:")
        for row in path[0].grid:
            print(" ".join(map(str, row)))
        print("--------")

        for state in path:
            print(f"Depth: {state.depth}")
            for row in state.grid:
                print(" ".join(map(str, row)))
            print("--------")

    def plot_path_seaborn(self, path):
        """Plots each state in the path using seaborn heatmaps."""
        for idx, state in enumerate(path):
            plt.figure(figsize=(3,3))
            sns.heatmap(state.grid, annot=True, cbar=False, square=True,
                        linewidths=1, linecolor="black", cmap="Blues",
                        fmt="d", annot_kws={"size":16})
            plt.title(f"Depth {state.depth}")
            plt.show()


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    initial_state_dict = {0:[2,8,3], 1:[1,6,4], 2:[7,0,5]}
    goal_state_dict = {0:[1,2,3], 1:[8,0,4], 2:[7,6,5]}

    solver = PuzzleSolver(initial_state_dict, goal_state_dict)

    # Solve with BFS
    bfs_path = solver.solve_bfs()
    print("BFS Solution Path:")
    solver.print_path_text(bfs_path)
    solver.plot_path_seaborn(bfs_path)

    # Solve with DFS
    dfs_path = solver.solve_dfs(max_depth=30)
    print("DFS Solution Path:")
    solver.print_path_text(dfs_path)
    solver.plot_path_seaborn(dfs_path)
