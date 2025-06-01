import numpy as np
from d3rlpy.dataset import Episode, MDPDataset

def episodes_to_mdp_arrays(episode_list: list[Episode]):
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_timeouts = [] 

    for i, episode in enumerate(episode_list):
        if len(episode) == 0:
            print(f"Warning: Skipping empty episode at index {i}.")
            continue

        all_observations.append(episode.observations)
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)

        # Create terminal flags for this episode's data
        terminals = np.zeros(len(episode), dtype=np.float32)
        if len(episode) > 0: # Ensure episode is not empty before accessing -1
            terminals[-1] = 1.0  # Only the last step of the episode is terminal
        all_terminals.append(terminals)

        # Timeouts: d3rlpy uses this to distinguish between actual termination
        # and termination due to time limit. For offline data where 'done'
        # means end of session, timeouts are usually all False (0.0).
        timeouts = np.zeros(len(episode), dtype=np.float32)
        all_timeouts.append(timeouts)

    try:
        print("Concatenating episode data arrays...")
        final_observations = np.concatenate(all_observations, axis=0)
        final_actions = np.concatenate(all_actions, axis=0)
        final_rewards = np.concatenate(all_rewards, axis=0)
        final_terminals = np.concatenate(all_terminals, axis=0)
        final_timeouts = np.concatenate(all_timeouts, axis=0) # Added timeouts
        print("Concatenation complete.")
        return final_observations, final_actions, final_rewards, final_terminals, final_timeouts
    except ValueError as e:
        print(f"Error during array concatenation: {e}")
        print("This might happen if episodes have inconsistent data shapes or types.")
        return None, None, None, None, None