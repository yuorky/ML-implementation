from rl_provided import *
import numpy as np
from typing import Tuple, List
from copy import deepcopy


def get_transition_prob(n_sa, n_sas, curr_state: State, dir_intended: int, next_state: State) -> float:
    """
    Determine the transition probability based on counts in n_sa and n_sas'.
    curr_state is s. dir_intended is a. next_state is s'.

    n_sa is 3d, (x,y,a) where x is row, y is column and a is action
    dir_intended: up:0, right:1, down:2, left:3

    @return: 0 if we haven't visited the state-action pair yet (i.e. N_sa = 0).
      Otherwise, return N_sas' / N_sa.
    """
    N_sa = n_sa[curr_state][dir_intended]
    N_sas = n_sas[curr_state][dir_intended][next_state]

    if N_sa == 0:
        return 0
    else:
        return N_sas/N_sa


def exp_utils(world, utils, n_sa, n_sas, curr_state: State) -> List[float]:
    """
    @return: The expected utility values for all four possible actions.
    i.e. calculates sum_s'( P(s' | s, a) * U(s')) for all four possible actions.

    The returned list contains the expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the expected utility
    if the agent ends up going down from the current state.
    """

    exp_utils_for_a = []
    next_states = set(get_next_states(world.grid, curr_state))
    for a in range(4):
        utils_a = 0
        for state in next_states:
            p = get_transition_prob(n_sa, n_sas, curr_state, a, state)
            u = utils[state]
            utils_a += p*u

        exp_utils_for_a.append(utils_a)

    return exp_utils_for_a


def optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> List[float]:
    """
    @return: The optimistic expected utility values for all four possible actions.
    i.e. calculates f ( sum_s'( P(s' | s, a) * U(s')), N(s, a) ) for all four possible actions.

    The returned list contains the optimistic expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the optimistic expected utility
    if the agent ends up going down from the current state.
    """
    exp_utils_for_a = exp_utils(world, utils, n_sa, n_sas, curr_state)
    opt_utils_for_a = [0] * world.num_actions
    for a in range(4):
        N_sa = n_sa[curr_state][a]
        if N_sa < n_e:
            opt_utils_for_a[a]=r_plus
        else:
            opt_utils_for_a[a]= exp_utils_for_a[a]
    return opt_utils_for_a


def update_utils(world, utils, n_sa, n_sas, n_e: int, r_plus: float) -> np.ndarray:
    """
    Update the utility values via value iteration until they converge.
    Call `utils_converged` to check for convergence.
    @return: The updated utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    new_util = np.zeros(world.grid.shape)
    i = 0
    while i == 0 or not utils_converged(utils, new_util):
        if i > 0:
            utils = deepcopy(new_util)
        i += 1
        for r in range(world.grid.shape[0]):
            for c in range(world.grid.shape[1]):
                cur_state = State((r, c))
                if is_wall(world.grid, cur_state):
                    new_util[cur_state] = 0
                    continue
                if is_goal(world.grid, cur_state):
                    new_util[cur_state] = world.grid[cur_state]
                    continue
                opt_utils_for_a = optimistic_exp_utils(world, utils, n_sa, n_sas, cur_state, n_e, r_plus)
                opt_a = np.argmax(opt_utils_for_a)
                value_new = world.reward + world.discount * opt_utils_for_a[opt_a]
                new_util[cur_state] = value_new

    return new_util


def get_best_action(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> int:
    """
    @return: The best action, based on the agent's current understanding of the world, to perform in `curr_state`.
    """
    opt_utils_for_a = optimistic_exp_utils(world,utils, n_sa,n_sas,curr_state,n_e,r_plus)
    return int(np.argmax(opt_utils_for_a))


def adpa_move(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> Tuple[State, np.ndarray]:
    """
    Execute ADP for one move. This function performs the following steps.
        1. Choose best action based on the utility values (utils).
        2. Make a move by calling `make_move_det`.
        3. Update the counts.
        4. Update the utility values (utils) via value iteration.
        5. Return the new state and the new utilities.

    @return: The state the agent ends up in after performing what it thinks is the best action + the updated
      utilities after performing this action.
    @rtype: A tuple (next_state, next_utils), where:
     - next_utils is an `np.ndarray` of size `world.grid.shape` of type `float`
    """
    action = get_best_action(world, utils, n_sa, n_sas, curr_state, n_e, r_plus)
    next_state = world.make_move_det(action, n_sa)

    n_sa[curr_state][action] += 1
    n_sas[curr_state][action][next_state] += 1

    new_utils = update_utils(world, utils, n_sa, n_sas, n_e, r_plus)

    return next_state, new_utils


def utils_to_policy(world, utils, n_sa, n_sas) -> np.ndarray:
    """
    @return: The optimal policy derived from the given utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `int`.
    """
    # Initialize the policy.
    policy = np.zeros(world.grid.shape, dtype=int)

    for r in range(world.grid.shape[0]):
        for c in range(world.grid.shape[1]):
            cur_state = State((r, c))
            if is_wall(world.grid, cur_state):
                continue
            if is_goal(world.grid, cur_state):
                continue
            exp_util = exp_utils(world, utils, n_sa, n_sas, cur_state)
            opt_a = np.argmax(exp_util)
            policy[cur_state] = int(opt_a)

    return policy


def is_done_exploring(n_sa, grid, n_e: int) -> bool:
    """
    @return: True when the agent has visited each state-action pair at least `n_e` times.
    """
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cur_state = State((r,c))
            for d in range(4):
                if n_sa[cur_state][d] < n_e and not is_goal(grid, cur_state) and not is_wall(grid, cur_state):
                    return False

    return True


def adpa(world_name: str, n_e: int, r_plus: float) -> np.ndarray:
    """
    Perform active ADP. Runs a certain number of moves and returns the learned utilities and policy.
    Stops when the agent is done exploring the world and the utility values have converged.
    Call `utils_converged` to check for convergence.

    Note: By convention, our tests expect the utility of a "wall" state to be 0.

    @param world_name: The name of the world we wish to explore.
    @param n_e: The minimum number of times we wish to see each state-action pair.
    @param r_plus: The maximum reward we can expect to receive in any state.
    @return: The learned utilities.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    # Initialize the world
    world = read_world(world_name)
    grid = world.grid
    num_actions = world.num_actions

    # Initialize persistent variable
    utils = np.zeros(grid.shape)
    n_sa = np.zeros((*grid.shape, num_actions))
    n_sas = np.zeros((*grid.shape, num_actions, *grid.shape))

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if is_goal(grid, State((r,c))):
                utils[(r,c)] = grid[(r,c)]

    converged = False

    while not is_done_exploring(n_sa, grid, n_e) or not converged:

        next_state, new_utils = adpa_move(world, utils, n_sa, n_sas, world.curr_state, n_e, r_plus)
        if utils_converged(utils, new_utils):
            converged = True
        utils = deepcopy(new_utils)

    return utils


# cstate = State((1, 2))
# lec_world = read_world('lecture')
#
# n_sa_test = np.array([[[3., 3., 3., 3.],
#                        [3., 3., 3., 3.],
#                        [3., 3., 3., 3.],
#                        [3., 3., 3., 3.]],
#                       [[3., 3., 3., 3.],
#                        [0., 0., 0., 0.],
#                        [2., 3., 3., 3.],
#                        [3., 0., 0., 0.]],
#                       [[3., 3., 3., 3.],
#                        [3., 3., 3., 3.],
#                        [3., 3., 3., 3.],
#                        [0., 0., 0., 0.]]])
#
# adpa('lecture', 30, 1)
#
# world = read_world('lecture')
# grid = world.grid
# num_actions = world.num_actions
#
# # Initialize persistent variable
# utils = np.zeros(grid.shape)
# for r in range(grid.shape[0]):
#     for c in range( grid.shape[1] ):
#         if is_goal( grid, State( (r, c) ) ):
#             utils[(r, c)] = grid[(r, c)]
# n_sa = np.zeros((*grid.shape, num_actions))
# n_sas = np.zeros((*grid.shape, num_actions, *grid.shape))
# curr_state = world.curr_state
#
# #converged = False
# while not is_done_exploring(n_sa, grid, 1):
#     next_state, new_utils = adpa_move(world, utils, n_sa, n_sas, curr_state, 1, -0.04)
#     # if utils_converged(utils, new_utils):
#     #     converged = True
#     utils = new_utils
#     curr_state = next_state
