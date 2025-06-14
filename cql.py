import os
import pickle
import d3rlpy.logging as d3rlpy_logging
import d3rlpy.metrics as d3rlpy_metrics
from d3rlpy.dataset import Episode, MDPDataset
from d3rlpy.algos import DiscreteCQLConfig
from eval_cql import evaluate_policy_cql
import numpy as np
from d3rlpy import LoggingStrategy
from utils import episodes_to_mdp_arrays

# data prams
DATA_DIR = "data"
TRAIN_EPISODES_PATH = os.path.join(DATA_DIR, "train_episodes.pkl")
TEST_EPISODES_PATH = os.path.join(DATA_DIR, "test_episodes.pkl")

# save params
MODEL_SAVE_DIR = "trained_models"
LOG_DIR = "logs"
EXPERIMENT_NAME = "EcommerceCQL_v1"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# train params
GAMMA = 0.99 # Discount factor
BATCH_SIZE = 256
ACTOR_LR=1e-4,  # policy (actor) lr
CRITIC_LR=3e-4, # Q-functions (critic) lr
INITIAL_ALPHA = 1.0, # Initial value for alpha (Lagrangian multiplier)
ALPHA_THRESHOLD = 10.0, # If (logsumexp Q - E[Q]) > threshold, alpha increases.
CONSERVATIVE_WEIGHT = 5.0, # Conservatism strength
N_TRAIN_STEPS = 200000

# evaluation params
FQE_LEARNING_RATE = 1e-4
FQE_N_STEPS = 100000 
FQE_BATCH_SIZE = BATCH_SIZE

DEVICE = 'cuda:0'
# DEVICE = 'cpu'
    
def train_cql(train_episodes_list: list[Episode], test_episodes_list: list[Episode], 
              model_save_path: str, experiment_name: str, logdir: str):    

    cql_config = DiscreteCQLConfig(
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        # learning_rate=ACTOR_LR,       
        # critic_learning_rate=CRITIC_LR,
        # initial_alpha=INITIAL_ALPHA,            
        # alpha_threshold=ALPHA_THRESHOLD,            
        # conservative_weight=CONSERVATIVE_WEIGHT,            
        # n_action_samples=10,    # Number of actions to sample for CQL term
        # action_scaler=d3rlpy.preprocessing.MinMaxActionScaler() # If actions were continuous
        # observation_scaler=d3rlpy.preprocessing.StandardScaler() # If observations need scaling
        # q_func_factory=d3rlpy.models.q_functions.MeanQFunctionFactory(), # Default is fine for now
        # actor_encoder_factory=d3rlpy.models.encoders.DefaultEncoderFactory(), # Default MLP is fine
    )

    policy = cql_config.create(device=DEVICE)

    observations, actions, rewards, terminals, timeouts = episodes_to_mdp_arrays(train_episodes_list)
    mdp_train_dataset = MDPDataset(
        observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                timeouts=timeouts)
    
    observations_test, actions_test, rewards_test, terminals_test, timeouts_test = episodes_to_mdp_arrays(test_episodes_list)
    mdp_test_dataset = MDPDataset(
        observations=observations_test,
                actions=actions_test,
                rewards=rewards_test,
                terminals=terminals_test,
                timeouts=timeouts_test)
    
    evaluators={
        'td_error': d3rlpy_metrics.TDErrorEvaluator(mdp_test_dataset.episodes),
        'average_q_value': d3rlpy_metrics.AverageValueEstimationEvaluator(mdp_test_dataset.episodes),
        'initial_state_value': d3rlpy_metrics.InitialStateValueEstimationEvaluator(mdp_test_dataset.episodes),
    }

    logger_adapter = d3rlpy_logging.CombineAdapterFactory([
        d3rlpy_logging.FileAdapterFactory(root_dir=logdir),
        d3rlpy_logging.TensorboardAdapterFactory(root_dir=logdir),
        # d3rlpy_logging.WanDBAdapterFactory(project="RL_ecommerce"),
    ])
    
    try:
        policy.build_with_dataset(mdp_train_dataset)
    except Exception as e:
        print(f"Error during CQL policy.build_with_dataset(): {e}")
        return None
    
    policy.fit(
        mdp_train_dataset,
        n_steps=N_TRAIN_STEPS,
        experiment_name=experiment_name,
        logger_adapter=logger_adapter,
        logging_steps = 500,
        logging_strategy = LoggingStrategy.STEPS,
        evaluators=evaluators,
    )

    # Save the trained model
    print(f"Saving trained cql model to: {model_save_path}")
    policy.save_model(model_save_path)
    return policy


try:
    with open(TRAIN_EPISODES_PATH, 'rb') as f:
        train_episodes = pickle.load(f)
    if not train_episodes:
        print("ERROR: Training episodes list is empty.")
        exit()
except Exception as e:
    print(f"Error loading training episodes: {e}")
    exit()

try:
    with open(TEST_EPISODES_PATH, 'rb') as f:
        test_episodes = pickle.load(f)
    if not test_episodes:
        print("ERROR: Test episodes list is empty.")
        exit()
except Exception as e:
    print(f"Error loading test episodes: {e}")
    exit()


cql_model_save_path = os.path.join(MODEL_SAVE_DIR, "cql_policy.pt")
log_dir = os.path.join(LOG_DIR, EXPERIMENT_NAME)

train_cql(
    train_episodes_list=train_episodes,
    test_episodes_list=test_episodes,
    model_save_path=cql_model_save_path,
    experiment_name=EXPERIMENT_NAME,
    logdir=log_dir
)
