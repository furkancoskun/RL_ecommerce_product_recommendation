from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.dataset import Episode, MDPDataset
import d3rlpy.metrics as d3rlpy_metrics
import d3rlpy.logging as d3rlpy_logging
import numpy as np
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig 
from utils import episodes_to_mdp_arrays
from d3rlpy import LoggingStrategy

LOG_DIR = "logs"
EXPERIMENT_NAME = "DiscreteFQE_EvalBC"

def evaluate_policy_bc(
    policy_model_path: str,
    train_episodes_list: list[Episode],
    test_episodes_list: list[Episode],
    gamma: float,
    device: str,
    fqe_n_steps: int,
    fqe_lr: float,
    fqe_batch_size: int,
):
    try:
        loaded_policy_config = DiscreteBCConfig(gamma=gamma)
        policy_to_evaluate = loaded_policy_config.create(device=device)

        observations, actions, rewards, terminals, timeouts = episodes_to_mdp_arrays(train_episodes_list)
        mdp_train_dataset = MDPDataset(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                timeouts=timeouts)
        policy_to_evaluate.build_with_dataset(mdp_train_dataset)

        observations_test, actions_test, rewards_test, terminals_test, timeouts_test = episodes_to_mdp_arrays(test_episodes_list)
        mdp_test_dataset = MDPDataset(
        observations=observations_test,
                actions=actions_test,
                rewards=rewards_test,
                terminals=terminals_test,
                timeouts=timeouts_test)
            
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
    fqe_estimator = DiscreteFQE(algo=policy_to_evaluate, config=fqe_config, device=device)
    
    logger_adapter = d3rlpy_logging.CombineAdapterFactory([
        d3rlpy_logging.FileAdapterFactory(root_dir=LOG_DIR),
        d3rlpy_logging.TensorboardAdapterFactory(root_dir=LOG_DIR),
        # d3rlpy_logging.WanDBAdapterFactory(project="RL_ecommerce"),
    ])

    print(f"Training FQE model for {fqe_n_steps} steps...")
    fqe_estimator.fit(
        mdp_test_dataset,
        n_steps=fqe_n_steps,
        evaluators={
            'init_value': d3rlpy_metrics.InitialStateValueEstimationEvaluator(),
            'soft_opc': d3rlpy_metrics.SoftOPCEvaluator(return_threshold=-300),
        },
        logger_adapter=logger_adapter,
        logging_steps = 500,
        logging_strategy = LoggingStrategy.STEPS,
        experiment_name=EXPERIMENT_NAME,

    )
    return average_actual_return, std_actual_return


if __name__ == "__main__":
    import os
    import pickle

    MODEL_SAVE_DIR = "trained_models"
    POLICY_PATH = os.path.join(MODEL_SAVE_DIR, "bc_policy.pt")
    DATA_DIR = "data"
    TRAIN_EPISODES_PATH = os.path.join(DATA_DIR, "train_episodes.pkl")
    TEST_EPISODES_PATH = os.path.join(DATA_DIR, "test_episodes.pkl")
    GAMMA = 0.99
    FQE_LEARNING_RATE = 1e-4
    FQE_N_STEPS = 100000
    FQE_BATCH_SIZE = 128
    DEVICE = 'cuda:0' # 'cpu'
    
    with open(TEST_EPISODES_PATH, 'rb') as f:
        test_episodes = pickle.load(f)

    with open(TRAIN_EPISODES_PATH, 'rb') as f:
        train_episodes = pickle.load(f)

    print(f"POLICY PATH: {POLICY_PATH}")

    average_actual_return, std_actual_return = evaluate_policy_bc(
        POLICY_PATH,
        train_episodes,
        test_episodes,
        GAMMA,
        DEVICE,
        FQE_N_STEPS,
        FQE_LEARNING_RATE,
        FQE_BATCH_SIZE
    )

    print(f"Average Actual Return: {average_actual_return}")
    print(f"Standard Deviation of Actual Return: {std_actual_return}")
    