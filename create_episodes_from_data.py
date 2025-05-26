import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
from d3rlpy.dataset import Episode, MDPDataset
import os
import pickle

# PART-1 
input_filename = 'data/ecommerce_clickstream_transactions_filtered.csv'
save_dir = "data/"
os.makedirs(save_dir, exist_ok=True)
df = pd.read_csv(input_filename)

# Convert Timestamp to datetime objects if not already
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Sort primarily by user, then session, then time
df = df.sort_values(by=['UserID', 'SessionID', 'Timestamp']).reset_index(drop=True)
print(df.head())


# PART-2: Map IDs
unique_users = df['UserID'].unique()
user_id_map = {id: i for i, id in enumerate(unique_users)}
df['UserIndex'] = df['UserID'].map(user_id_map)

unique_products = df['ProductID'].unique()
# Reserve 0 for a potential 'start' or 'padding' token if needed later
product_id_map = {id: i+1 for i, id in enumerate(unique_products)}
# Add a mapping for a 'start' product (no previous product)
product_id_map['START'] = 0
df['ProductIndex'] = df['ProductID'].map(product_id_map)

n_users = len(unique_users)
n_products = len(unique_products) + 1 # +1 for START token
print(f"Mapped {n_users} users and {n_products} products (including START).")


# PART-3: Map Event Types
event_type_map = {'product_view': 0, 'add_to_cart': 1, 'purchase': 2, 'START': 3}
df['EventTypeIndex'] = df['EventType'].map(event_type_map)
n_event_types = len(event_type_map)


# PART-4: Define Rewards
reward_map = {
    'product_view': 0.1, # Low reward for engagement
    'add_to_cart': 0.5,  # Medium reward
    'purchase': 1.0,   # Highest reward
    'START': 0.0       # No reward for starting
}
# Note: Reward is assigned based on the event *at the end* of the transition


# PART-5:  Extract Transitions (Create the MDP dataset)
print("Grouping transitions by session...")
sessions_data = defaultdict(list)
grouped = df.groupby(['UserIndex', 'SessionID'])

for (user_idx, session_id), session_df in grouped:
    session_key = (user_idx, session_id)
    # Sort within the group just to be safe
    session_df = session_df.sort_values(by='Timestamp')

    last_product_idx = product_id_map['START']
    last_event_type_idx = event_type_map['START']

    for i in range(len(session_df)):
        current_event = session_df.iloc[i]
        state = (user_idx, last_product_idx, last_event_type_idx)
        action = current_event['ProductIndex']
        reward = reward_map[current_event['EventType']]
        next_state = (user_idx, current_event['ProductIndex'], current_event['EventTypeIndex'])
        done = (i == len(session_df) - 1)

        transition = {
            'state': np.array(state, dtype=np.float32), 
            'action': np.array(action, dtype=np.int32),
            'reward': np.array(reward, dtype=np.float32),
            'next_state': np.array(next_state, dtype=np.float32),
            'terminal': np.array(float(done), dtype=np.float32) 
        }
        sessions_data[session_key].append(transition)

        last_product_idx = current_event['ProductIndex']
        last_event_type_idx = current_event['EventTypeIndex']

print(f"Grouped data into {len(sessions_data)} sessions.")


# PART-5: Split into Train/Test
session_keys = list(sessions_data.keys())
# user_indices_for_stratification = [key[0] for key in session_keys] # Extract UserIndex from key

test_size = 0.05 # e.g., 5% for testing
random_state = 42 # for reproducibility

print(f"Splitting sessions into train/test ({1-test_size:.0%}/{test_size:.0%})...")
train_session_keys, test_session_keys = train_test_split(
    session_keys,
    test_size=test_size,
    # stratify=user_indices_for_stratification, # Stratify by UserIndex
    random_state=random_state,

)

print(f"Training sessions: {len(train_session_keys)}")
print(f"Test sessions: {len(test_session_keys)}")


# --- Step 6: Create Final Datasets (e.g., for d3rlpy) ---
def create_episode_list(session_keys, all_sessions_data):
    """Helper function to create a list of Episode objects."""
    episodes = []
    for key in session_keys:
        transitions = all_sessions_data.get(key)
        if not transitions: # Check if transitions exist for the session key
             print(f"Warning: Session key {key} not found or empty in sessions_data.")
             continue

        observations = np.array([t['state'] for t in transitions], dtype=np.float32)
        actions = np.array([t['action'] for t in transitions], dtype=np.int32) # Actions are discrete indices
        rewards = np.array([t['reward'] for t in transitions], dtype=np.float32)

        n_steps = len(observations)
        if n_steps == 0: # Handle cases where session might exist but have 0 valid transitions processed
            print(f"Warning: Session key {key} resulted in 0 transitions.")
            continue

        # Create terminated flags: 0.0 for non-terminal, 1.0 for the last step
        terminated = np.zeros(n_steps, dtype=np.float32)
        terminated[-1] = 1.0 # Mark the very last step as terminal for the episode

        try:
            episode = Episode(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminated=terminated
            )
            episodes.append(episode)
        except TypeError as e:
            print(f"Error creating Episode for key {key}: {e}")
            print(f"Shapes: obs={observations.shape}, act={actions.shape}, rew={rewards.shape}, term={terminated.shape}")
        except Exception as e:
             print(f"Unexpected error for key {key}: {e}")
    return episodes

print("Creating train dataset episodes...")
train_episodes = create_episode_list(train_session_keys, sessions_data)

print("Creating test dataset episodes...")
test_episodes = create_episode_list(test_session_keys, sessions_data)


# --- Step 7: Save the episodes ---
train_episodes_path_pkl = os.path.join(save_dir, "train_episodes.pkl")
test_episodes_path_pkl = os.path.join(save_dir, "test_episodes.pkl")

# Save training episodes
if train_episodes: # Check if the list is not empty
    print(f"Saving training episodes ({len(train_episodes)} episodes) to {train_episodes_path_pkl} using pickle...")
    try:
        with open(train_episodes_path_pkl, 'wb') as f:
            pickle.dump(train_episodes, f)
        print("Training episodes saved successfully (pickle).")
    except Exception as e:
        print(f"Error saving training episodes with pickle: {e}")
else:
    print("Skipping saving of training episodes as the list is empty.")

# Save test episodes
if test_episodes: # Check if the list is not empty
    print(f"Saving test episodes ({len(test_episodes)} episodes) to {test_episodes_path_pkl} using pickle...")
    try:
        with open(test_episodes_path_pkl, 'wb') as f:
            pickle.dump(test_episodes, f)
        print("Test episodes saved successfully (pickle).")
    except Exception as e:
        print(f"Error saving test episodes with pickle: {e}")
else:
    print("Skipping saving of test episodes as the list is empty.")