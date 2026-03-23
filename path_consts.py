import os
import json
from utils import get_repo_basedir

REPO_BASE_DIR = get_repo_basedir()

# Directories for Q learning experiments and welfare analysis of trajectories
EXPERIMENTS_DIR = os.path.join(REPO_BASE_DIR, "experiments")
ANALYSIS_RESULTS_DIR = os.path.join(REPO_BASE_DIR, "analysis_results")

# Trajectory data directories
TRAJECTORIES_DIR = os.path.join(REPO_BASE_DIR, "trajectories")
INTERACTIVE_TRAJECTORIES_DIR = os.path.join(REPO_BASE_DIR, "interactive_trajectories")

# User data directories
USER_DATA_DIR = os.path.join(REPO_BASE_DIR, "user_data")
USER_ANALYSIS_DIR = os.path.join(REPO_BASE_DIR, "user_data_analysis")

### Current raw data

# Pilot scenarios, within trajectories dir
CURRENT_SCENARIOS_DIR = os.path.join(TRAJECTORIES_DIR, "pilot_scenarios_v3")

# User responses to pilot scenarios, within user data dir
CURRENT_USER_DATA_FILE = os.path.join(USER_DATA_DIR, "pilot_data_v5.csv")
###


### Current processed/intermediate data

# Computed welfare reductions on pilot scenarios, within welfare analysis dir
#CURRENT_WELFARES_DIR = os.path.join(ANALYSIS_RESULTS_DIR, 'bulk_pilot_scenarios_v3__run2')
CURRENT_WELFARES_DIR = os.path.join(ANALYSIS_RESULTS_DIR, 'bulk_pilot_scenarios_v3__run4')

# Computed welfare reductions aligned with user responses to pilot scenarios, 
# CURRENT_ALIGNED_USER_DATA_DIR = os.path.join(USER_ANALYSIS_DIR, 'user_data_aligned_with_welfare_analysis__from_pilotv5_run2')
CURRENT_ALIGNED_USER_DATA_DIR = os.path.join(USER_ANALYSIS_DIR, 'user_data_aligned_with_welfare_analysis__from_pilotv5_run5')

### 

from consts import HumanPolicies, NaoSupportPolicies, RewardTypes, GameTypes, MinimalComplexityHumanPolicies


# class ExperimentRegistry:
#     def __init__(self):
#         self.experiments = {}

#     def register_experiment(self, experiment_name, experiment_dir, model_path):
#         self.experiments[experiment_name] = (experiment_dir, model_path)

#     def get_experiment(self, experiment_name):
#         return self.experiments[experiment_name]
    
# class Experiment:
#     def __init__(self, experiment_di):
#         self.experiment_name = experiment_name
#         self.experiment_dir = experiment_dir
#         self.model_path = model_path

def validate_model_path(experiment_dir, model_path, game_type: GameTypes, human_policy: HumanPolicies, nao_support_policy: NaoSupportPolicies, reward_model: RewardTypes, minimal_complexity_env: bool=False, minimal_complexity_human_policy: MinimalComplexityHumanPolicies=None, verbose: bool=False):
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # check if model_path is inside experiment_dir
    if not os.path.commonpath([experiment_dir, model_path]) == experiment_dir:
        raise OSError(f"Model path is not inside experiment directory: {model_path} is not in {experiment_dir}")
    
    # check if args used to run the experiment match desired model parameters
    with open(os.path.join(experiment_dir, "args.json"), 'r') as file:
        experiment_args = json.load(file)
        # TODO? use enums instead of strings
        if experiment_args["game_type"] != game_type.value:
            raise ValueError(f"Game type mismatch: looking for {game_type.value} but experiment trained with {experiment_args['game_type']}")
        if experiment_args["human_policy"] != human_policy.value:
            raise ValueError(f"Human policy mismatch: looking for {human_policy.value} but experiment trained with {experiment_args['human_policy']}")
        if experiment_args["support_policy"] != nao_support_policy.value:
            raise ValueError(f"Support policy mismatch: looking for {nao_support_policy.value} but experiment trained with {experiment_args['support_policy']}")
        if experiment_args["reward_model"] != reward_model.value:
            raise ValueError(f"Reward model mismatch: looking for {reward_model.value} but experiment trained with {experiment_args['reward_model']}")
        if experiment_args["minimal_complexity_env"] != minimal_complexity_env:
            raise ValueError(f"Minimal complexity env mismatch: looking for {minimal_complexity_env} but experiment trained with {experiment_args['minimal_complexity_env']}")
        if experiment_args["minimal_complexity_human_policy"] != minimal_complexity_human_policy.value:
            raise ValueError(f"Minimal complexity human policy mismatch: looking for {minimal_complexity_human_policy.value} but experiment trained with {experiment_args['minimal_complexity_human_policy']}")
    if verbose:
        print(f"Loaded model from experiment directory: {experiment_dir}, model path: {model_path}")
        if os.path.exists(os.path.join(experiment_dir, "description.txt")):
            with open(os.path.join(experiment_dir, "description.txt"), 'r') as file:
                description = file.read()
                print(f"Description: {description}")
    
def get_manual_model_path(experiment_name):
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    model_dir = os.path.join(experiment_dir, "models")
    model_filenames = os.listdir(model_dir)
    if len(model_filenames) != 1:
        raise FileNotFoundError(f"Expected 1 model file in {model_dir}, found {model_filenames}")
    model_path = os.path.join(model_dir, model_filenames[0])
    return experiment_dir, model_path

def get_model_path(game_type: GameTypes, human_policy: HumanPolicies, nao_support_policy: NaoSupportPolicies, reward_model: RewardTypes, minimal_complexity_env: bool=False, minimal_complexity_human_policy: MinimalComplexityHumanPolicies=None, verbose=False):
    
    experiment_dir = None
    model_path = None
    if game_type == GameTypes.COMPETITIVE:
        if human_policy == HumanPolicies.RULES_BASED and nao_support_policy == NaoSupportPolicies.EQUAL_SUPPORT and reward_model == RewardTypes.FULL:
            experiment_dir = os.path.join(EXPERIMENTS_DIR, "v3_nao=equal__human=rules__actionspace=human")
            model_path = os.path.join(experiment_dir, "models/env=competitive__support=equalSupport__reward=full.pt")
            
    if experiment_dir is None or model_path is None:
        raise ValueError(f"Invalid experiment configuration: game_type={game_type}, human_policy={human_policy}, nao_support_policy={nao_support_policy}, reward_model={reward_model}")
    
    validate_model_path(experiment_dir, model_path, game_type, human_policy, nao_support_policy, reward_model, minimal_complexity_env, minimal_complexity_human_policy, verbose)
    
    return experiment_dir, model_path
    