"""
8-Puzzle Solver using BFS and DFS Algorithms

This module provides an object-oriented solution to the 8-puzzle problem,
where numbered tiles must be arranged in clockwise ascending order.
"""

from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class PuzzleState:
    """
    Represents a single state of the 8-puzzle.
    
    Attributes:
        state_dict (dict): Dictionary representation of the puzzle grid
        blank_pos (tuple): Position of the blank cell (row, col)
        depth (int): Depth of this state in the search tree
        parent (PuzzleState): Reference to the parent state
        move (str): The move that led to this state
    """
    
    def __init__(self, state_dict, depth=0, parent=None, move=None):
        """
        Initialize a puzzle state.
        
        Args:
            state_dict (dict): Dictionary with keys 0,1,2 representing rows
            depth (int): Current depth in search tree
            parent (PuzzleState): Parent state that led to this state
            move (str): Direction moved to reach this state
        """
        self.state_dict = state_dict
        self.depth = depth
        self.parent = parent
        self.move = move
        self.blank_pos = self._find_blank()
    
    def _find_blank(self):
        """
        Find the position of the blank cell (0) in the puzzle.
        
        Returns:
            tuple: (row, col) position of the blank cell
        """
        for row in range(3):
            for col in range(3):
                if self.state_dict[row][col] == 0:
                    return (row, col)
        return None
    
    def get_state_tuple(self):
        """
        Convert state to a tuple for hashing and comparison.
        This allows us to track visited states efficiently.
        
        Returns:
            tuple: Immutable representation of the puzzle state
        """
        return tuple(tuple(self.state_dict[i]) for i in range(3))
    
    def is_goal(self, goal_state):
        """
        Check if this state matches the goal state.
        
        Args:
            goal_state (PuzzleState): The goal state to compare against
            
        Returns:
            bool: True if this is the goal state, False otherwise
        """
        return self.get_state_tuple() == goal_state.get_state_tuple()
    
    def get_possible_moves(self):
        """
        Generate all valid moves from the current state.
        The blank cell can move up, down, left, or right if within bounds.
        
        Returns:
            list: List of direction strings ('up', 'down', 'left', 'right')
        """
        moves = []
        row, col = self.blank_pos
        
        # Check each direction and add if within grid bounds
        if row > 0:  # Can move up (swap with cell above)
            moves.append('up')
        if row < 2:  # Can move down (swap with cell below)
            moves.append('down')
        if col > 0:  # Can move left (swap with cell to the left)
            moves.append('left')
        if col < 2:  # Can move right (swap with cell to the right)
            moves.append('right')
        
        return moves
    
    def apply_move(self, move):
        """
        Apply a move and generate a new state.
        Moving the blank "up" means swapping it with the tile above it.
        
        Args:
            move (str): Direction to move ('up', 'down', 'left', 'right')
            
        Returns:
            PuzzleState: New state after applying the move
        """
        # Create a deep copy of the current state
        new_state_dict = {i: self.state_dict[i][:] for i in range(3)}
        row, col = self.blank_pos
        
        # Determine the new position based on the move direction
        if move == 'up':
            new_row, new_col = row - 1, col
        elif move == 'down':
            new_row, new_col = row + 1, col
        elif move == 'left':
            new_row, new_col = row, col - 1
        elif move == 'right':
            new_row, new_col = row, col + 1
        
        # Swap the blank with the target cell
        new_state_dict[row][col] = new_state_dict[new_row][new_col]
        new_state_dict[new_row][new_col] = 0
        
        # Create and return the new state
        return PuzzleState(new_state_dict, self.depth + 1, self, move)
    
    def __str__(self):
        """
        String representation of the puzzle state for printing.
        
        Returns:
            str: Formatted grid representation
        """
        result = []
        for row in range(3):
            result.append(' '.join(str(self.state_dict[row][col]) 
                                   for col in range(3)))
        return '\n'.join(result)


class PuzzleSolver:
    """
    Solver class for the 8-puzzle problem.
    Implements both BFS and DFS search algorithms.
    """
    
    def __init__(self, initial_state_dict, goal_state_dict):
        """
        Initialize the puzzle solver.
        
        Args:
            initial_state_dict (dict): Starting configuration
            goal_state_dict (dict): Target configuration
        """
        self.initial_state = PuzzleState(initial_state_dict)
        self.goal_state = PuzzleState(goal_state_dict)
        self.solution_path = []  # Will store the path from start to goal
        self.all_states = []  # Will store all states explored
    
    def solve_bfs(self):
        """
        Solve the puzzle using Breadth-First Search (BFS).
        BFS explores all states at depth d before moving to depth d+1.
        This guarantees finding the shortest solution path.
        
        Returns:
            bool: True if solution found, False otherwise
        """
        # Queue for BFS: process states in FIFO order
        queue = deque([self.initial_state])
        
        # Set to track visited states (using tuple representation)
        visited = {self.initial_state.get_state_tuple()}
        
        # List to store all explored states for visualization
        self.all_states = [self.initial_state]
        
        while queue:
            # Get the next state to explore (from front of queue)
            current_state = queue.popleft()
            
            # Check if we've reached the goal
            if current_state.is_goal(self.goal_state):
                self._reconstruct_path(current_state)
                return True
            
            # Generate all possible next states
            for move in current_state.get_possible_moves():
                next_state = current_state.apply_move(move)
                state_tuple = next_state.get_state_tuple()
                
                # Only process unvisited states
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    queue.append(next_state)
                    self.all_states.append(next_state)
        
        # No solution found
        return False
    
    def solve_dfs(self, max_depth=50):
        """
        Solve the puzzle using Depth-First Search (DFS).
        DFS explores as far as possible along each branch before backtracking.
        Note: DFS may not find the shortest path and can get stuck in deep branches.
        
        Args:
            max_depth (int): Maximum depth to explore (prevents infinite loops)
            
        Returns:
            bool: True if solution found, False otherwise
        """
        # Stack for DFS: process states in LIFO order
        stack = [self.initial_state]
        
        # Set to track visited states
        visited = {self.initial_state.get_state_tuple()}
        
        # List to store all explored states
        self.all_states = [self.initial_state]
        
        while stack:
            # Get the next state to explore (from top of stack)
            current_state = stack.pop()
            
            # Check if we've reached the goal
            if current_state.is_goal(self.goal_state):
                self._reconstruct_path(current_state)
                return True
            
            # Only explore further if we haven't exceeded max depth
            if current_state.depth < max_depth:
                # Generate all possible next states
                for move in current_state.get_possible_moves():
                    next_state = current_state.apply_move(move)
                    state_tuple = next_state.get_state_tuple()
                    
                    # Only process unvisited states
                    if state_tuple not in visited:
                        visited.add(state_tuple)
                        stack.append(next_state)
                        self.all_states.append(next_state)
        
        # No solution found within max_depth
        return False
    
    def _reconstruct_path(self, goal_state):
        """
        Reconstruct the solution path by following parent pointers.
        This works backwards from the goal state to the initial state.
        
        Args:
            goal_state (PuzzleState): The goal state node
        """
        path = []
        current = goal_state
        
        # Traverse backwards through parent pointers
        while current is not None:
            path.append(current)
            current = current.parent
        
        # Reverse to get path from initial to goal
        self.solution_path = list(reversed(path))
    
    def print_solution(self):
        """
        Print the solution path in text format.
        Shows each state with its depth and the move that led to it.
        """
        if not self.solution_path:
            print("No solution found!")
            return
        
        print("=" * 40)
        print("SOLUTION PATH")
        print("=" * 40)
        
        for i, state in enumerate(self.solution_path):
            if i == 0:
                print("Initial State:")
            else:
                print(f"\nMove: {state.move}")
            
            print(f"Depth: {state.depth}")
            print(state)
            print("-" * 20)
        
        print(f"\nTotal moves: {len(self.solution_path) - 1}")
        print(f"Total states explored: {len(self.all_states)}")
    
    def print_all_states(self):
        """
        Print all explored states in text format.
        This shows the complete search process, not just the solution path.
        """
        if not self.all_states:
            print("No states to display!")
            return
        
        print("=" * 40)
        print("ALL EXPLORED STATES")
        print("=" * 40)
        
        for i, state in enumerate(self.all_states):
            print(f"\nState #{i + 1}")
            print(f"Depth: {state.depth}")
            if state.move:
                print(f"Move: {state.move}")
            print(state)
            print("-" * 20)
        
        print(f"\nTotal states explored: {len(self.all_states)}")
    
    def visualize_solution(self):
        """
        Visualize the solution path using Seaborn heatmaps.
        Each state is shown as a colored grid with numbers.
        """
        if not self.solution_path:
            print("No solution to visualize!")
            return
        
        n_states = len(self.solution_path)
        
        # Calculate grid layout for subplots
        # Arrange in rows with up to 5 columns
        cols = min(5, n_states)
        rows = (n_states + cols - 1) // cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        fig.suptitle('Solution Path Visualization', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easy iteration
        if n_states == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each state
        for idx, state in enumerate(self.solution_path):
            # Convert state to numpy array for heatmap
            grid = np.array([state.state_dict[i] for i in range(3)])
            
            # Create heatmap
            sns.heatmap(grid, annot=True, fmt='d', cmap='YlOrRd', 
                       cbar=False, square=True, linewidths=2,
                       linecolor='black', ax=axes[idx],
                       annot_kws={'size': 16, 'weight': 'bold'})
            
            # Set title
            if idx == 0:
                title = 'Initial State'
            elif idx == len(self.solution_path) - 1:
                title = 'Goal State'
            else:
                title = f'Step {idx}: {state.move}'
            
            axes[idx].set_title(f"{title}\nDepth: {state.depth}", 
                               fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_states, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_all_states(self, max_states=20):
        """
        Visualize all explored states (or a subset) using Seaborn heatmaps.
        Limited to max_states to avoid overwhelming visualizations.
        
        Args:
            max_states (int): Maximum number of states to display
        """
        if not self.all_states:
            print("No states to visualize!")
            return
        
        # Limit the number of states to display
        states_to_show = self.all_states[:max_states]
        n_states = len(states_to_show)
        
        # Calculate grid layout
        cols = min(5, n_states)
        rows = (n_states + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        fig.suptitle(f'All Explored States (showing {n_states} of {len(self.all_states)})', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes
        if n_states == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each state
        for idx, state in enumerate(states_to_show):
            grid = np.array([state.state_dict[i] for i in range(3)])
            
            sns.heatmap(grid, annot=True, fmt='d', cmap='viridis', 
                       cbar=False, square=True, linewidths=2,
                       linecolor='black', ax=axes[idx],
                       annot_kws={'size': 14, 'weight': 'bold'})
            
            axes[idx].set_title(f"State {idx+1}\nDepth: {state.depth}", 
                               fontsize=9)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_states, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXAMPLE USAGE (Ready for Jupyter Notebook)
# ============================================================================

if __name__ == "__main__":
    # Define initial and goal states as specified
    initial_state_dict = {0: [2, 8, 3], 1: [1, 6, 4], 2: [7, 0, 5]}
    goal_state_dict = {0: [1, 2, 3], 1: [8, 0, 4], 2: [7, 6, 5]}
    
    print("8-PUZZLE SOLVER DEMONSTRATION")
    print("=" * 60)
    
    # ========== BFS SOLUTION ==========
    print("\n\n*** BREADTH-FIRST SEARCH (BFS) ***\n")
    
    solver_bfs = PuzzleSolver(initial_state_dict, goal_state_dict)
    
    if solver_bfs.solve_bfs():
        print("Solution found using BFS!")
        solver_bfs.print_solution()
        print("\n")
        solver_bfs.visualize_solution()
    else:
        print("No solution found using BFS.")
    
    # ========== DFS SOLUTION ==========
    print("\n\n*** DEPTH-FIRST SEARCH (DFS) ***\n")
    
    solver_dfs = PuzzleSolver(initial_state_dict, goal_state_dict)
    
    if solver_dfs.solve_dfs():
        print("Solution found using DFS!")
        solver_dfs.print_solution()
        print("\n")
        solver_dfs.visualize_solution()
    else:
        print("No solution found using DFS.")