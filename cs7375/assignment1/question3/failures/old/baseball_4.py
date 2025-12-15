"""
eight_puzzle_solver.py — Recursive DFS version

Object-oriented solver for the 8-puzzle problem.

Implements:
- solve_bfs(): Breadth-First Search
- solve_dfs(): Depth-First Search (recursive)
- print_states_text() / print_solution_text()
- print_states_graphic(): visualizes states using Seaborn

All states (initial, intermediate, final) are created and recorded
until the goal is reached.

Compatible for import into a Jupyter Notebook or standard Python script.
"""

from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import copy
import seaborn as sns
import matplotlib.pyplot as plt


class EightPuzzleSolver:
    def __init__(self, initial_state_dict: Dict[int, List[int]], goal_state_dict: Dict[int, List[int]]):
        self.initial_state = self._dict_to_tuple(initial_state_dict)
        self.goal_state = self._dict_to_tuple(goal_state_dict)

        self.all_states_created: List[Tuple[Tuple[int, ...], int]] = []
        self.parent: Dict[Tuple[int, ...], Optional[Tuple[int, ...]]] = {}
        self.depth_map: Dict[Tuple[int, ...], int] = {}
        self.solution_path: List[Tuple[int, ...]] = []
        self.goal_found: bool = False  # flag for recursive DFS

    # ---------------------------
    # Conversion helpers
    # ---------------------------
    @staticmethod
    def _dict_to_tuple(d: Dict[int, List[int]]) -> Tuple[int, ...]:
        rows = []
        for r in range(3):
            rows.extend(d[r])
        return tuple(rows)

    @staticmethod
    def _tuple_to_rows(state: Tuple[int, ...]) -> List[List[int]]:
        return [list(state[0:3]), list(state[3:6]), list(state[6:9])]

    @staticmethod
    def _pretty_print_state(state: Tuple[int, ...]) -> str:
        rows = EightPuzzleSolver._tuple_to_rows(state)
        return "\n".join(" ".join(str(x) for x in r) for r in rows)

    # ---------------------------
    # Neighbor generation
    # ---------------------------
    def get_neighbors(self, state: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        zero_idx = state.index(0)
        r, c = divmod(zero_idx, 3)
        neighbors = []

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                ni = nr * 3 + nc
                lst = list(state)
                lst[zero_idx], lst[ni] = lst[ni], lst[zero_idx]
                neighbors.append(tuple(lst))
        return neighbors

    # ---------------------------
    # Solvability check
    # ---------------------------
    @staticmethod
    def _inversion_count(state: Tuple[int, ...]) -> int:
        arr = [x for x in state if x != 0]
        return sum(arr[i] > arr[j] for i in range(len(arr)) for j in range(i + 1, len(arr)))

    def is_solvable(self) -> bool:
        return (self._inversion_count(self.initial_state) % 2) == (
            self._inversion_count(self.goal_state) % 2
        )

    # ---------------------------
    # Breadth-First Search (same as before)
    # ---------------------------
    def solve_bfs(self, max_nodes: Optional[int] = None):
        self.all_states_created = []
        self.parent = {}
        self.depth_map = {}
        self.solution_path = []

        if not self.is_solvable():
            raise ValueError("Puzzle is not solvable.")

        start = self.initial_state
        goal = self.goal_state

        queue = deque([start])
        self.parent[start] = None
        self.depth_map[start] = 0
        self.all_states_created.append((start, 0))
        visited: Set[Tuple[int, ...]] = {start}

        nodes_created = 1

        while queue:
            current = queue.popleft()
            depth = self.depth_map[current]

            if current == goal:
                self._reconstruct_solution(goal)
                return

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.parent[neighbor] = current
                    self.depth_map[neighbor] = depth + 1
                    self.all_states_created.append((neighbor, depth + 1))
                    queue.append(neighbor)
                    nodes_created += 1
                    if max_nodes and nodes_created >= max_nodes:
                        return

    # ---------------------------
    # Depth-First Search (recursive)
    # ---------------------------
    def solve_dfs(self, max_depth: Optional[int] = None):
        """
        Recursive Depth-First Search version.

        Args:
            max_depth: optional recursion limit (prevents infinite recursion).
        """
        self.all_states_created = []
        self.parent = {}
        self.depth_map = {}
        self.solution_path = []
        self.goal_found = False

        if not self.is_solvable():
            raise ValueError("Puzzle is not solvable.")

        start = self.initial_state
        goal = self.goal_state

        self.parent[start] = None
        self.depth_map[start] = 0
        self.all_states_created.append((start, 0))

        visited = set()

        def dfs_recursive(state: Tuple[int, ...], depth: int):
            # Stop recursion if goal already found
            if self.goal_found:
                return
            visited.add(state)

            if state == goal:
                self.goal_found = True
                self._reconstruct_solution(goal)
                return

            if max_depth is not None and depth >= max_depth:
                return

            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    self.parent[neighbor] = state
                    self.depth_map[neighbor] = depth + 1
                    self.all_states_created.append((neighbor, depth + 1))
                    dfs_recursive(neighbor, depth + 1)
                    if self.goal_found:
                        return  # early exit after finding goal

        dfs_recursive(start, 0)

    # ---------------------------
    # Solution reconstruction
    # ---------------------------
    def _reconstruct_solution(self, goal_state: Tuple[int, ...]):
        path = []
        cur = goal_state
        while cur is not None:
            path.append(cur)
            cur = self.parent.get(cur)
        path.reverse()
        self.solution_path = path

    # ---------------------------
    # Printing & visualization
    # ---------------------------
    def print_states_text(self, states: Optional[List[Tuple[Tuple[int, ...], int]]] = None):
        if states is None:
            states = self.all_states_created
        if not states:
            print("(no states to print)")
            return

        print("Initial State:")
        print(self._pretty_print_state(self.initial_state))
        print("--------")

        for state, depth in states:
            print(f"Depth: {depth}")
            print(self._pretty_print_state(state))
            print("--------")

    def print_solution_text(self):
        if not self.solution_path:
            print("No solution path found.")
            return
        print("Solution Path:")
        for i, state in enumerate(self.solution_path):
            print(f"Depth: {i}")
            print(self._pretty_print_state(state))
            print("--------")

    def print_states_graphic(
        self,
        states: Optional[List[Tuple[Tuple[int, ...], int]]] = None,
        cols: int = 4,
        figsize_per_cell: Tuple[int, int] = (2, 2),
    ):
        if states is None:
            states = self.all_states_created
        if not states:
            print("(no states to visualize)")
            return

        total = len(states)
        cols = max(1, cols)
        rows = (total + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_cell[0], rows * figsize_per_cell[1]))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, (state, depth) in enumerate(states):
            grid = self._tuple_to_rows(state)
            sns.heatmap(grid, annot=True, fmt="d", cbar=False, square=True, linewidths=0.5, ax=axes[i])
            axes[i].set_title(f"Depth {depth}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    initial_state_dict = {0: [2, 8, 3], 1: [1, 6, 4], 2: [7, 0, 5]}
    goal_state_dict = {0: [1, 2, 3], 1: [8, 0, 4], 2: [7, 6, 5]}

    solver = EightPuzzleSolver(initial_state_dict, goal_state_dict)

    if not solver.is_solvable():
        print("Puzzle not solvable.")
    else:
        print("=== Solving with recursive DFS ===")
        solver.solve_dfs(max_depth=25)
        print(f"Total states discovered: {len(solver.all_states_created)}")
        solver.print_solution_text()
        path_states_with_depth = [(s, solver.depth_map.get(s, i)) for i, s in enumerate(solver.solution_path)]
        solver.print_states_graphic(path_states_with_depth, cols=4)
