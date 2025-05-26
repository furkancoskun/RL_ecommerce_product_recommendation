from d3rlpy.ope import FQEConfig, FQE
from d3rlpy.dataset import Episode 
import numpy as np

def evaluate_policy(
    policy_model_path: str,
    PolicyAlgoConfig,
    test_episodes_list: list[Episode],
    gamma: float,
    device: str,
    fqe_n_steps: int,
    fqe_lr: float,
    fqe_batch_size: int,
):
    try:
        loaded_policy_config = PolicyAlgoConfig(gamma=gamma)
        policy_to_evaluate = loaded_policy_config.create(device=device)
        policy_to_evaluate.load_model(policy_model_path)
    except Exception as e:
        print(f"Error loading trained policy from {policy_model_path}: {e}")
        return None, None, None

    # Calculate Average Actual Return (Behavior Policy Baseline on Test Set)
    actual_returns = []
    for episode in test_episodes_list:
        cumulative_reward = 0.0
        for i, reward in enumerate(episode.rewards):
            cumulative_reward += (gamma ** i) * reward
        actual_returns.append(cumulative_reward)
    
    average_actual_return = np.mean(actual_returns) if actual_returns else 0.0
    std_actual_return = np.std(actual_returns) if actual_returns else 0.0
    print(f"Average Actual Return (Behavior Policy on Test Set): {average_actual_return:.4f} +/- {std_actual_return:.4f}")

    fqe_config = FQEConfig(
        learning_rate=fqe_lr,
        batch_size=fqe_batch_size,
        gamma=gamma
    )
    fqe_estimator = FQE(algo=policy_to_evaluate, config=fqe_config, device=device)
    
    print(f"Training FQE model for {fqe_n_steps} steps...")
    fqe_estimator.fit(
        test_episodes_list,
        n_steps=fqe_n_steps
    )

    estimated_value = fqe_estimator.estimate_policy_value(test_episodes_list)
    return estimated_value, average_actual_return, std_actual_return


def __main__():
    import os
    import pickle
    from d3rlpy.algos import CQLConfig, IQLConfig

    MODEL_SAVE_DIR = "trained_models"
    POLICY_PATH = os.path.join(MODEL_SAVE_DIR, "cql_policy.pt") # "iql_policy.pt"
    DATA_DIR = "data"
    TEST_EPISODES_PATH = os.path.join(DATA_DIR, "test_episodes.pkl")
    GAMMA = 0.99
    FQE_LEARNING_RATE = 1e-4
    FQE_N_STEPS = 100000
    FQE_BATCH_SIZE = 128
    DEVICE = 'cuda' # 'cpu'
    
    with open(TEST_EPISODES_PATH, 'rb') as f:
        test_episodes = pickle.load(f)

    estimated_value, average_actual_return, std_actual_return = evaluate_policy(
        POLICY_PATH,
        CQLConfig, #IQLConfig
        test_episodes,
        GAMMA,
        DEVICE,
        FQE_N_STEPS,
        FQE_LEARNING_RATE,
        FQE_BATCH_SIZE
    )

    print(f"Estimated Value: {estimated_value}")
    print(f"Average Actual Return: {average_actual_return}")
    print(f"Standard Deviation of Actual Return: {std_actual_return}")
    