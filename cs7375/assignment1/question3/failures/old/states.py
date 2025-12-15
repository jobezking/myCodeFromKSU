from collections import deque
from numpy import np

initial_state = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]
final_goal_state = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
initial_state_dict = {0:[2,8,3], 1:[1,6,4], 2:[7,0,5]}
goal_state_dict = {0:[1,2,3], 1:[8,0,4], 2:[7,6,5]}


initial_state_np = np.array([[2, 8, 3], [1, 6, 4], [7, 0, 5]])
goal_state_np = np.array([1, 2, 3], [8, 0, 4], [7, 6, 5])