#
# 8-Puzzle Solver using BFS and DFS
#
# This object-oriented Python program solves the 8-puzzle problem. [cite: 1]
# It includes methods for both Breadth-First Search (BFS) and Depth-First Search (DFS)
# and provides two types of output: text-based and a graphical representation.
# The code is designed to be well-commented and easily usable in a Jupyter
# Notebook environment. [cite: 8, 9]
#

# Import necessary libraries
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Node Class to represent each state of the puzzle ---
# This class stores the state configuration, its parent node (to reconstruct the path),
# the action that led to this state, and its depth in the search tree.
class PuzzleNode:
    """A class to represent a state (node) in the 8-puzzle search tree."""
    def __init__(self, state, parent=None, action=None, depth=0):
        """
        Initializes a PuzzleNode.
        Args:
            state (tuple): A tuple of tuples representing the 3x3 grid.
            parent (PuzzleNode): The node that generated this node.
            action (str): The move ('Up', 'Down', 'Left', 'Right') that led to this state.
            depth (int): The depth of the node in the search tree.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    # The __eq__ and __hash__ methods are essential for using nodes in sets
    # (like a 'visited' set) to check if a state has already been explored.
    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

# --- 2. Main Solver Class ---
# This class encapsulates the logic for solving the 8-puzzle.
class EightPuzzleSolver:
    """
    An object-oriented class to solve the 8-puzzle problem using BFS and DFS. [cite: 1]
    """
    def __init__(self, initial_state_dict, goal_state_dict):
        """
        Initializes the solver with the initial and goal states. 
        Args:
            initial_state_dict (dict): A dictionary representing the start state. 
            goal_state_dict (dict): A dictionary representing the goal state. 
        """
        # Convert the dictionary format to a tuple of tuples, which is immutable and hashable.
        self.initial_state = self._dict_to_tuple(initial_state_dict)
        self.goal_state = self._dict_to_tuple(goal_state_dict)

    def _dict_to_tuple(self, state_dict):
        """Helper function to convert the dictionary representation to a tuple of tuples."""
        # Sorting by key ensures the rows are in the correct order (0, 1, 2).
        return tuple(tuple(state_dict[key]) for key in sorted(state_dict.keys()))

    def _find_blank(self, state):
        """
        Helper function to find the (row, col) coordinates of the blank tile (0).
        Args:
            state (tuple): A state tuple.
        Returns:
            A tuple (row, col) of the blank tile's position.
        """
        for r_idx, row in enumerate(state):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return (r_idx, c_idx)
        return None

    def _get_neighbors(self, node):
        """
        Generates all valid successor states (neighbors) from a given node.
        Args:
            node (PuzzleNode): The current PuzzleNode.
        Returns:
            A list of new PuzzleNode objects representing valid next states.
        """
        neighbors = []
        blank_row, blank_col = self._find_blank(node.state)
        possible_moves = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]

        for dr, dc, action in possible_moves:
            new_row, new_col = blank_row + dr, blank_col + dc

            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state_list = [list(row) for row in node.state]
                new_state_list[blank_row][blank_col], new_state_list[new_row][new_col] = \
                    new_state_list[new_row][new_col], new_state_list[blank_row][blank_col]
                new_state_tuple = tuple(tuple(row) for row in new_state_list)
                
                new_node = PuzzleNode(
                    state=new_state_tuple,
                    parent=node,
                    action=action,
                    depth=node.depth + 1
                )
                neighbors.append(new_node)
        return neighbors

    def _reconstruct_path(self, final_node):
        """
        Traces back from the goal node to the initial node using parent pointers
        to build the solution path. 
        Args:
            final_node (PuzzleNode): The goal node found by the search algorithm.
        Returns:
            A list of PuzzleNode objects from initial to goal state.
        """
        path = []
        current = final_node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    def solve_bfs(self):
        """
        Solves the 8-puzzle using the Breadth-First Search (BFS) algorithm. 
        BFS explores level by level, guaranteeing the shortest path in terms of moves.
        Returns:
            The solution path as a list of PuzzleNodes, or None if no solution is found.
        """
        frontier = collections.deque([PuzzleNode(self.initial_state, depth=0)])
        visited = {self.initial_state}
        
        while frontier:
            current_node = frontier.popleft()

            if current_node.state == self.goal_state:
                return self._reconstruct_path(current_node)
            
            for neighbor in self._get_neighbors(current_node):
                if neighbor.state not in visited:
                    visited.add(neighbor.state)
                    frontier.append(neighbor)
        
        return None

    def solve_dfs(self):
        """
        Solves the 8-puzzle using the Depth-First Search (DFS) algorithm. 
        DFS explores as deeply as possible along each branch before backtracking.
        It does not guarantee the shortest path.
        Returns:
            The solution path as a list of PuzzleNodes, or None if no solution is found.
        """
        frontier = [PuzzleNode(self.initial_state, depth=0)]
        visited = set()

        while frontier:
            current_node = frontier.pop()
            
            if current_node.state in visited:
                continue
            
            visited.add(current_node.state)
            
            if current_node.state == self.goal_state:
                return self._reconstruct_path(current_node)

            for neighbor in reversed(self._get_neighbors(current_node)):
                if neighbor.state not in visited:
                    frontier.append(neighbor)
        
        return None

# --- 3. Output Functions ---
# These functions are used to display the solution path. [cite: 6]
def print_text_solution(path):
    """
    Prints all states in the solution path in a text-based format. [cite: 6]
    Args:
        path (list): A list of PuzzleNode objects representing the solution path.
    """
    if path is None:
        print("No solution found.")
        return

    print("Initial State:")
    for row in path[0].state:
        print(' '.join(map(str, row)))
    print("--------")
    
    for node in path:
        print(f"Depth: {node.depth}")
        if node.action:
            print(f"Action: {node.action}")
        for row in node.state:
            print(' '.join(map(str, row)))
        print("--------")
    print(f"Solution found in {len(path)-1} moves.")

def plot_graphic_solution(path):
    """
    Creates a simple graphic representation of the solution path using Seaborn heatmaps. [cite: 7]
    Args:
        path (list): A list of PuzzleNode objects representing the solution path.
    """
    if path is None:
        print("No solution found to plot.")
        return

    num_steps = len(path)
    cols = min(num_steps, 5)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if num_steps == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()

    for i, node in enumerate(path):
        ax = axes[i]
        data = np.array(node.state)
        sns.heatmap(data, ax=ax, annot=True, cbar=False, cmap="viridis",
                    linewidths=1, linecolor='gray', fmt='d',
                    annot_kws={"size": 16})
        title = f"Initial (Depth {node.depth})" if i == 0 else f"Step {i} (Depth {node.depth})"
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(num_steps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# --- 4. Main execution block ---
# This block demonstrates how to use the solver when run as a standalone script.
if __name__ == '__main__':
    initial_state_dict = {0: [2, 8, 3], 1: [1, 6, 4], 2: [7, 0, 5]} [cite: 2]
    goal_state_dict = {0: [1, 2, 3], 1: [8, 0, 4], 2: [7, 6, 5]} [cite: 2]

    solver = EightPuzzleSolver(initial_state_dict, goal_state_dict)

    print("=========================")
    print("Solving with BFS...")
    print("=========================")
    bfs_path = solver.solve_bfs()
    print_text_solution(bfs_path)
    plot_graphic_solution(bfs_path)
    
    print("\n\n")

    print("=========================")
    print("Solving with DFS...")
    print("=========================")
    dfs_path = solver.solve_dfs()
    print_text_solution(dfs_path)
    plot_graphic_solution(dfs_path)


''' 
=================================================
How to Run the Program in a Jupyter Notebook 📝
=================================================

This program is structured as a Python script that can be easily imported and
used in a Jupyter Notebook to display the text and graphical outputs. [cite: 8, 9]

### Step 1: Save the Code
First, save the complete Python program above into a file named `puzzle_solver.py`.

### Step 2: Create and Run the Notebook
Create a new Jupyter Notebook (`.ipynb` file) in the same directory and add the
following code cells.

#### Cell 1: Import the Solver and Output Functions
This cell imports the necessary classes and functions from your `puzzle_solver.py` file.

```python
# Import the main solver class and the two output functions
from puzzle_solver import EightPuzzleSolver, print_text_solution, plot_graphic_solution
'''