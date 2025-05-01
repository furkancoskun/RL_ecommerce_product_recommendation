import pandas as pd

# PART-1 
input_filename = 'ecommerce_clickstream_transactions_filtered.csv'
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
transitions = []
# Group by user and session
grouped = df.groupby(['UserIndex', 'SessionID'])

for (user_idx, session_id), session_df in grouped:
    # Sort again just to be absolutely sure (should be redundant if Step 1 was done right)
    session_df = session_df.sort_values(by='Timestamp')
    
    # Initialize the 'last' state variables for the start of the session
    last_product_idx = product_id_map['START']
    last_event_type_idx = event_type_map['START']
    
    for i in range(len(session_df)):
        current_event = session_df.iloc[i]
        
        # State (s_t): Based on the *previous* event (or START)
        state = (user_idx, last_product_idx, last_event_type_idx)
        
        # Action (a_t): The product interacted with in the *current* event
        # In offline RL, the logged action is what the user *did*
        action = current_event['ProductIndex']
        
        # Reward (r_t): Based on the *current* event's type
        reward = reward_map[current_event['EventType']]
        
        # Next State (s_{t+1}): Based on the *current* event
        next_state = (user_idx, current_event['ProductIndex'], current_event['EventTypeIndex'])
        
        # Done flag: Is this the last event in the session?
        done = (i == len(session_df) - 1)
        
        # Store the transition
        transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            # Store original IDs too for potential debugging/analysis
            'user_id': current_event['UserID'],
            'product_id': current_event['ProductID'],
            'event_type': current_event['EventType'],
            'timestamp': current_event['Timestamp']
        })
        
        # Update 'last' variables for the next iteration
        last_product_idx = current_event['ProductIndex']
        last_event_type_idx = current_event['EventTypeIndex']

print(f"Extracted {len(transitions)} transitions.")
# Example: View the first few transitions
print(transitions[:3])


