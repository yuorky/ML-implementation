from board import *
from copy import deepcopy
from heapq import *


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    init_state = State(init_board, hfn, hfn(init_board), 0, parent=None)
    if is_goal(init_state):
        return [init_state], 0

    frontier = [(init_state.f, init_state.id, 0, init_state)]
    explored_states = set()

    while frontier:
        _, state_id, _, state_self = heappop(frontier)
        if state_id not in explored_states:
            explored_states.add(state_id)
            if is_goal(state_self):
                return get_path(state_self), state_self.depth

            for next_state in get_successors(state_self):
                heappush(frontier, (next_state.f, next_state.id, state_id, next_state))

    return [], -1


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    init_state = State(init_board, None, 0, 0, parent=None)
    if is_goal(init_state):
        return [init_state], 0

    frontier = [init_state]
    explored_states = set()

    while frontier:
        state_last = frontier.pop()
        if state_last.id not in explored_states:
            explored_states.add(state_last.id)
            if is_goal(state_last):
                cost = state_last.depth
                return get_path(state_last), cost

            frontier += sorted(get_successors(state_last), key=lambda x: x.id, reverse=True)

    return [], -1


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    state_last = []

    temp_board = state.board
    depth = state.depth
    directions = ['forward', 'backward']
    for i in range(len(temp_board.cars)):
        # move forward(right, down) or backward(left, up)
        for direction in directions:
            car = temp_board.cars[i]
            grid = temp_board.grid
            while is_movable(car, direction, grid):
                new_car = deepcopy(car)
                new_cars = deepcopy(temp_board.cars)
                new_car = move(new_car, direction)
                new_cars[i] = new_car
                new_board = Board(temp_board.name, temp_board.size, new_cars)
                if not state.hfn:
                    state_last.append(State(new_board, None, 0, depth+1, state))
                else:
                    state_last.append(State(new_board, state.hfn, state.hfn(new_board) + state.depth, depth+1, state))
                car = new_car
                grid = new_board.grid

    return state_last


def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    for car in state.board.cars:
        if car.is_goal and car.var_coord == 4:
            return True
        return False


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """

    state_path = [state]
    while state.parent:
        state_path.append(state.parent)
        state = state.parent

    return state_path[::-1]


def blocking_heuristic(board):
    """
    Returns the heuristic value for the given board
    based on the Blocking Heuristic function.

    Blocking heuristic returns zero at any goal board,
    and returns one plus the number of cars directly
    blocking the goal car in all other states.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    right_end = board.grid[2].index('>')
    if right_end + 1 == board.size:
        return 0
    else:
        h = 1
        for i in range(right_end + 1, board.size):
            if board.grid[2][i] != '.':
                h += 1
        return h


def advanced_heuristic(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


# determine if the car is movable in 1 grid range
def is_movable(car, direction, grid):
    """ determine if the car can move"""
    length = car.length
    if direction == 'forward':
        if car.orientation == 'h':
            col = car.var_coord
            row = car.fix_coord
            if col+length >= len(grid):
                return False
            if grid[row][col+length] != '.':
                return False

        if car.orientation == 'v':
            col = car.fix_coord
            row = car.var_coord
            if row + length >= len(grid):
                return False
            if grid[row+length][col] != '.':
                return False

    if direction == 'backward':
        if car.orientation == 'h':
            col = car.var_coord
            row = car.fix_coord
            if col <= 0:
                return False
            if grid[row][col-1] != '.':
                return False

        if car.orientation == 'v':
            col = car.fix_coord
            row = car.var_coord
            if row <= 0:
                return False
            if grid[row-1][col] != '.':
                return False

    return True


def move(car, direction):
    if direction == 'forward':  # col+1/row+1: right/down
        car.var_coord += 1
    if direction == 'backward':  # col-1/row-1: left/up
        car.var_coord -= 1

    return car

#
# aa = from_file('jams_posted.txt')
# board1 = aa[0]

# result = dfs(board1)
# result2 = dfs(aa[1])
# result3 = dfs(aa[2])
#
# a_result = a_star(board1,blocking_heuristic)
# a_result_2 = a_star(aa[1], blocking_heuristic)
# a_result_3 = a_star(aa[2], blocking_heuristic)
#
# pathHeu, costHeu = a_result
# for i, state in enumerate(pathHeu, 1):
#     print(i, "/", len(pathHeu), " ", state.f - state.depth, blocking_heuristic(state.board))
#     state.board.display()

# board2 = aa[1]
# board3 = aa[2]
# init_state = State(board1, None, 0, 0, parent=None)
# second_state = State(board2, None, 0, 1, parent=init_state)
# third_state = State(board3, None, 0, 2, parent=second_state)
# frontier = [[init_state]]
# path = frontier.pop()
# state_last = path[-1]
#
# # test get_path()
# path_3 = get_path(third_state)
#
# # test is_goal()
# goal = is_goal(third_state)  # False

