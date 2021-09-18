# version 1.0

import numpy as np
from typing import List, Dict

from utils_soln import *

def create_observation_matrix(env: Environment):
    '''
    Creates a 2D numpy array containing the observation probabilities for each state. 

    Entry (i,j) in the array is the probability of making an observation type j in state i.

    Saves the matrix in env.observe_matrix and returns nothing.

    P(Ot = j | Si)
    '''
    d_r, d_c = len(env.state_types), env.num_observe_types
    env.observe_matrix = np.zeros((d_r, d_c))

    for i, si in enumerate(env.state_types):
        env.observe_matrix[i] = env.observe_probs[si]


def create_transition_matrices(env: Environment):
    '''
    If the transition_matrices in env is not None, 
    constructs a 3D numpy array containing the transition matrix for each action.

    Entry (i,j,k) is the probability of transitioning from state j to k
    given that the agent takes action i.

    Saves the matrices in env.transition_matrices and returns nothing.
    '''

    if env.transition_matrices is not None:
        return
    d_i = len(env.action_effects)
    d_j = d_k = len(env.state_types)
    env.transition_matrices = np.zeros((d_i, d_j, d_k))

    for i in range(d_i):
        offsets = env.action_effects[i].keys()
        probs = list(env.action_effects[i].values())
        for j in range(d_j):
            cur_state = j
            pos = [None] * len(offsets)
            for l, offset in enumerate(offsets):
                pos[l] = (cur_state + offset) % d_j
            for k in range(d_k):
                if k in pos:
                    env.transition_matrices[i][j][k] = probs[pos.index(k)]
                else:
                    env.transition_matrices[i][j][k] = 0


def forward_recursion(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Perform the filtering task for all the time steps.

    Calculate and return the values f_{0:0} to f_{0,t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.
    :param probs_init: The initial probabilities over the N states. regarding to position

    :return: A numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the normalized values of f_{0:k} (0 <= k <= t - 1).
    '''

    create_observation_matrix(env)
    create_transition_matrices(env)

    t = len(observ)
    N = env.num_states
    forward_matrices = np.zeros((t, N))

    # get base case f0:0
    ob_0 = observ[0]
    for i in range(N):
        forward_matrices[0][i] = env.observe_matrix[i][ob_0] * probs_init[i]

    normalize(forward_matrices[0])
    # get recursive case
    prev = 0
    for action in actions:
        trans_m = np.dot(env.transition_matrices[action], forward_matrices[prev])
        for j in range(N):
            ob_j = observ[prev+1]
            forward_matrices[prev+1][j] = env.observe_matrix[j][ob_j] * trans_m[j]
        prev += 1
        normalize(forward_matrices[prev])

    return forward_matrices


def backward_recursion(env: Environment, actions: List[int], observ: List[int] \
    ) -> np.ndarray:
    '''
    Perform the smoothing task for each time step.

    Calculate and return the values b_{1:t-1} to b_{t:t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.

    :return: A numpy array with shape (t+1, N), (N is the number of states)
            the k'th row represents the values of b_{k:t-1} (1 <= k <= t - 1),
            while the k=0 row is meaningless and we will NOT test it.
    '''

    create_observation_matrix(env)
    create_transition_matrices(env)

    t = len(observ)
    N = env.num_states
    backward_matrices = np.zeros((t+1, N))

    # get the base case b t:(t-1)
    backward_matrices[-1] = np.ones(N)

    # get recursive case
    prev = len(observ)
    for action in actions[::-1]:
        for j in range(N):
            for k in range(N):
                ob_j = observ[prev-1]
                backward_matrices[prev - 1][j] += env.observe_matrix[k][ob_j] * backward_matrices[prev][k] *\
                                                  env.transition_matrices[action][j][k]
        prev -= 1

    return backward_matrices


def fba(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Execute the forward-backward algorithm. 

    Calculate and return a 2D numpy array with shape (t,N) where t = len(observ) and N is the number of states.
    The k'th row represents the smoothed probability distribution over all the states at time step k.

    :param env: The environment.
    :param actions: A list of agent's past actions.
    :param observ: A list of observations.
    :param probs_init: The agent's initial beliefs over states
    :return: A numpy array with shape (t, N)
        the k'th row represents the normalized smoothed probability distribution over all the states for time k.
    '''

    t = len(observ)
    N = env.num_states
    fba_matrices = np.zeros((t, N))
    forward_m = forward_recursion(env, actions, observ, probs_init)
    backward_m = backward_recursion(env, actions, observ)
    for i in range(t):
        fba_matrices[i] = np.multiply(forward_m[i], backward_m[i+1])
        normalize(fba_matrices[i])
    return fba_matrices


def normalize(forward_matrices_row):
    row_sum = sum(forward_matrices_row)
    for i, val in enumerate(forward_matrices_row):
        forward_matrices_row[i] = val/row_sum


ue_num_state_types = 2
ue_state_types = [0, 1]
ue_num_observe_types = 2
ue_observe_probs = [[0.9, 0.1], [0.2, 0.8]]
ue_transition_matrices = np.array([[[0.7, 0.3], [0.3, 0.7]]])

env = Environment(ue_num_state_types, ue_state_types,
                  ue_num_observe_types, ue_observe_probs,
                  None, ue_transition_matrices)

ue_init_probs = [1. / env.num_states] * env.num_states
#
# visualize_belief(env, ue_init_probs)
#
# rb_num_state_types = 2
# rb_state_types = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
# rb_num_observe_types = 2
# rb_observe_probs = [[0.8, 0.2], [0.1, 0.9]]
#
# rb_action_effects = [None]*3
# rb_action_effects[0] = {0: 0.1, -1: 0.8, -2: 0.1}
# rb_action_effects[1] = {0: 0.1, 1: 0.8, 2: 0.1}
# rb_action_effects[2] = {0: 1.0}
#
# env_rb = Environment(rb_num_state_types, rb_state_types,
#                      rb_num_observe_types, rb_observe_probs,
#                      rb_action_effects, None)
#
# forward_recursion(env_rb, [0,1], [0,1,0], [1/16]*16)
#backward_recursion(env_rb, [0,1], [0,1,0])

# a
# state_types = [0,1]
# num_state_types = 2
# num_observe_types = 2
# observe_probs = [[0.1, 0.9], [0.8, 0.2]]
# action_effects = [None] * 3
# action_effects[0] = {0:0.05, -1:0.95}
# action_effects[1] = {0:0.15, 1:0.85}
# action_effects[2] = {0:1.0}
# env = Environment(num_state_types, state_types,num_observe_types,observe_probs,action_effects,None)
#
# actions = [1,0]
# observ = [1,0,0]
# init = [1/2, 1/2]
# print(forward_recursion(env, actions,observ, init))
# print(fba(env, actions, observ, init))
#
# actions_c = [1,0]
# observ = [0,1]