import numpy as np
import copy
from itertools import chain


def get_states_actions_next_states(states, actions, xp=np):
    # Prepare demonstrations
    # deep copy is necessary because demo_states can be list of lists
    next_states = copy.deepcopy(states)

    if states.ndim > 2:
        # demo_states.shape = (n_episode, n_steps, *observation.shape)
        # if each episode in demo has same length, demo_states and demo_actions will be numpy array
        # delete last state and action because there is no next state of last state
        states = states[:, :-1, ...]
        actions = actions[:, :-1, ...]
        # delete first state to make demo_next_states[:, i, ...] be the next states of the demo_states[:, i, ...]
        next_states = next_states[:, 1:, ...]
    else:
        # if length of episodes are different, delete last state, action, and first state
        for demo_states_epi, demo_action_epi, demo_next_state_epi \
                in zip(states, actions, next_states):
            # delete last state and action because there is no next state of last state
            del demo_states_epi[-1]
            del demo_action_epi[-1]
            # delete first state to make demo_next_states_epi[i] be the next states of the demo_states_epi[i]
            del demo_next_state_epi[0]

    states = xp.asarray(np.array(list(chain(*states))).astype(dtype=np.float32))
    next_states = xp.asarray(np.array(list(chain(*next_states))).astype(dtype=np.float32))
    actions = xp.asarray(np.array(list(chain(*actions))).astype(dtype=np.float32))
    return states, actions, next_states