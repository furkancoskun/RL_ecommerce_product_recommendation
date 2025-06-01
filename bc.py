import os
import pickle
import d3rlpy.logging as d3rlpy_logging
import d3rlpy.metrics as d3rlpy_metrics
from d3rlpy.dataset import Episode, MDPDataset
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig 
from eval_bc import evaluate_policy_bc 
from d3rlpy import LoggingStrategy
from utils import episodes_to_mdp_arrays

# data prams
DATA_DIR = "data"
TRAIN_EPISODES_PATH = os.path.join(DATA_DIR, "train_episodes.pkl")
TEST_EPISODES_PATH = os.path.join(DATA_DIR, "test_episodes.pkl")

# save params
MODEL_SAVE_DIR = "trained_models"
LOG_DIR = "logs"
EXPERIMENT_NAME = "EcommerceBC_v1"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# train params
GAMMA = 0.99 # Discount factor
BATCH_SIZE = 256
BC_LR = 3e-4  # policy (actor) lr
N_TRAIN_STEPS = 200000

# evaluation params
FQE_LEARNING_RATE = 1e-4
FQE_N_STEPS = 100000 
FQE_BATCH_SIZE = BATCH_SIZE

DEVICE = 'cuda:0'
# DEVICE = 'cpu'
    
def train_bc(train_episodes_list: list[Episode], test_episodes_list: list[Episode], 
              model_save_path: str, experiment_name: str, logdir: str):    

    bc_config = DiscreteBCConfig(
        learning_rate=BC_LR,
        batch_size=BATCH_SIZE,
    )

    policy = bc_config.create(device=DEVICE)

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
        'imitation_accuracy_on_test': d3rlpy_metrics.DiscreteActionMatchEvaluator(mdp_test_dataset.episodes),
    }

    logger_adapter = d3rlpy_logging.CombineAdapterFactory([
        d3rlpy_logging.FileAdapterFactory(root_dir=logdir),
        d3rlpy_logging.TensorboardAdapterFactory(root_dir=logdir),
        # d3rlpy_logging.WanDBAdapterFactory(project="RL_ecommerce"),
    ])
    
    try:
        policy.build_with_dataset(mdp_train_dataset)
    except Exception as e:
        print(f"Error during BC policy.build_with_dataset(): {e}")
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
    print(f"Saving trained bc model to: {model_save_path}")
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


bc_model_save_path = os.path.join(MODEL_SAVE_DIR, "bc_policy.pt")
bc_log_dir = os.path.join(LOG_DIR, EXPERIMENT_NAME)

trained_bc_policy = train_bc(
    train_episodes_list=train_episodes,
    test_episodes_list=test_episodes,
    model_save_path=bc_model_save_path,
    experiment_name=EXPERIMENT_NAME,
    logdir=bc_log_dir
)
