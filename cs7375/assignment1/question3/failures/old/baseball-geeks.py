import seaborn as sns
import matplotlib.pyplot as plt
import copy

# Puzzle dimensions
N = 3

# Moves: Left, Right, Up, Down
row_moves = [0, 0, -1, 1]
col_moves = [-1, 1, 0, 0]

# Initial and Goal states
initial_state = [[2, 8, 3],
                 [1, 6, 4],
                 [7, 0, 5]]

goal_state = [[1, 2, 3],
              [8, 0, 4],
              [7, 6, 5]]

# Utility: check if move is valid
def is_valid(x, y):
    return 0 <= x < N and 0 <= y < N

# Utility: check if goal reached
def is_goal(board):
    return board == goal_state

# Utility: find blank (0) position
def find_blank(board):
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                return i, j
    return None

# ----------- OUTPUT FUNCTIONS --------------

def print_states_text(states, algo_name):
    print(f"\n--- {algo_name} Solution Path (Text) ---")
    for depth, state in enumerate(states):
        print(f"Depth {depth}:")
        for row in state:
            print(" ".join(map(str, row)))
        print("--------")

def print_states_graphic(states, algo_name):
    print(f"\n--- {algo_name} Solution Path (Graphics) ---")
    for depth, state in enumerate(states):
        plt.figure(figsize=(3,3))
        sns.heatmap(state, annot=True, cbar=False, square=True,
                    linewidths=1, linecolor="black", cmap="Blues",
                    fmt="d", annot_kws={"size":16})
        plt.title(f"{algo_name} - Depth {depth}")
        plt.show()

# ----------- RECURSIVE DFS -----------------

def dfs_recursive(board, visited, path, solutions):
    if is_goal(board):
        solutions.append(path.copy())
        return True

    x, y = find_blank(board)
    for i in range(4):
        new_x, new_y = x + row_moves[i], y + col_moves[i]
        if is_valid(new_x, new_y):
            new_board = copy.deepcopy(board)
            new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
            board_tuple = tuple(map(tuple, new_board))
            if board_tuple not in visited:
                visited.add(board_tuple)
                path.append(new_board)
                if dfs_recursive(new_board, visited, path, solutions):
                    return True
                path.pop()
    return False

def solve_puzzle_dfs(start):
    visited = {tuple(map(tuple, start))}
    path = [start]
    solutions = []
    dfs_recursive(start, visited, path, solutions)
    return solutions[0] if solutions else []

# ----------- RECURSIVE BFS -----------------

def bfs_recursive(queue, visited, solutions):
    if not queue:
        return False

    board, path = queue.pop(0)
    if is_goal(board):
        solutions.append(path)
        return True

    x, y = find_blank(board)
    for i in range(4):
        new_x, new_y = x + row_moves[i], y + col_moves[i]
        if is_valid(new_x, new_y):
            new_board = copy.deepcopy(board)
            new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
            board_tuple = tuple(map(tuple, new_board))
            if board_tuple not in visited:
                visited.add(board_tuple)
                queue.append((new_board, path + [new_board]))

    return bfs_recursive(queue, visited, solutions)

def solve_puzzle_bfs(start):
    visited = {tuple(map(tuple, start))}
    queue = [(start, [start])]
    solutions = []
    bfs_recursive(queue, visited, solutions)
    return solutions[0] if solutions else []

# ----------- DRIVER -----------------

if __name__ == "__main__":
    print("Initial State:")
    for row in initial_state:
        print(" ".join(map(str, row)))
    print("Goal State:")
    for row in goal_state:
        print(" ".join(map(str, row)))
    print("====================================")

    # DFS
    dfs_solution = solve_puzzle_dfs(initial_state)
    print_states_text(dfs_solution, "DFS")
    print_states_graphic(dfs_solution, "DFS")

    # BFS
    bfs_solution = solve_puzzle_bfs(initial_state)
    print_states_text(bfs_solution, "BFS")
    print_states_graphic(bfs_solution, "BFS")
