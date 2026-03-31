import git
import os
import sys
import json
import shutil
import datetime
import pygame
import gymnasium as gym
from gymnasium.envs.registration import register, registry
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from consts import GameTypes, ActionSpaces, RewardTypes, RenderModes, NaoSupportPolicies, InteractiveModes, Players, DynamicsConsts, ScoreConsts
import importlib
from enum import Enum
from agents.policies import Action
from agents.robot_integration import RobotActionRequest
from typing import List, Optional
#from interactive_gameplay.utils import GameScriptEvent # NOTE: moved to inside CustomJSONDecoder to avoid circular import

### Debugging utilities
def debughook(etype, value, tb):
    import pdb
    import traceback

    traceback.print_exception(etype, value, tb)
    pdb.pm()  # post-mortem debugger. Note: this drops pdb into the exception throwing context

def enable_debug_hook():
    sys.excepthook = debughook
### End of debugging utilities


### Repo/file management utilities
def get_git_branch_and_hash():
    repo = git.Repo(search_parent_directories=True)
    branch = f"branch: {str(repo.head.reference)}"
    sha = f"commit: {repo.head.object.hexsha}"

    unstaged_files = [item.a_path for item in repo.index.diff(None)]
    unstaged_files_str = f"unstaged changes in files: {unstaged_files}" if unstaged_files else "no unstaged changes"
    return branch, sha, unstaged_files_str

def human_readable_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_output_dir(dir_path, wipe_dir=False, ask_to_append_timestamp=False):
    """
    Create the output directory if it does not exist. Optionally wipe the directory if it already exists.

    Args:
        dir_path (str): The path to the directory to create
        wipe_dir (bool): Whether to delete the directory if it already exists. User will always be prompted before deletion.    
    """
    if os.path.exists(dir_path):
        if ask_to_append_timestamp:
            timestamp = human_readable_timestamp()
            user_input = input(
                f"Output dir \"{dir_path}\" already exists. Append timestamp \"{timestamp}\" to directory name? Press Enter to append, 'q' to quit ..."
            )
            if user_input.lower() == "q":
                print("Exiting...")
                sys.exit()
            else:
                dir_path = dir_path + "_" + timestamp

    if wipe_dir and os.path.exists(dir_path):
        user_input = input(
            f"Wipe output dir \"{dir_path}\" before running? Press Enter to continue, 'q' to quit ..."
        )
        if user_input.lower() == "q":
            print("Exiting...")
            sys.exit()
        else:
            shutil.rmtree(dir_path)
    
    os.makedirs(dir_path, exist_ok=False)
    return dir_path

def save_command_info(exp_dir, args=None):
    print('\n'.join(get_git_branch_and_hash()), file=open(os.path.join(exp_dir, 'git_hash.txt'), 'w'))
    print('python ' + ' '.join(sys.argv), file=open(os.path.join(exp_dir, 'command.txt'), 'w'))
    print('timestamp: ' + human_readable_timestamp(), file=open(os.path.join(exp_dir, 'timestamp.txt'), 'w'))
    if args is not None:
        json.dump(vars(args), open(os.path.join(exp_dir, 'args.json'), 'w'))
        if "exp_description" in args and args.exp_description is not None:
            with open(os.path.join(exp_dir, "description.txt"), "w") as f:
                f.write(args.exp_description)

def init_experiment_dir(experiments_basedir, args, wipe_dir=False):
    if args.exp_name is None:
        executable_file = os.path.basename(sys.argv[0])
        exp_dir = os.path.join(experiments_basedir, f"{executable_file}:{human_readable_timestamp()}")
    else:
        exp_dir = os.path.join(experiments_basedir, args.exp_name)

    create_output_dir(exp_dir, wipe_dir=wipe_dir)
    save_command_info(exp_dir, args)
    return exp_dir

def get_repo_basedir():
    # Get the root directory of the git repo
    repo = git.Repo(search_parent_directories=True)
    return repo.working_dir

def get_active_venv_path(name_only=False):
    # Get the name of the active virtual environment, if any
    venv_path = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv_path is not None:
        if name_only:
            return os.path.basename(venv_path)
        else:
            return venv_path
    return None

### End of repo/file management utilities


### Environment utilities
class EnvRenderWrapper(gym.Wrapper):
    def __init__(self, env, render=False, framerate=None, record_frames=True):
        super(EnvRenderWrapper, self).__init__(env)
        self.render = render
        self.framerate = framerate
        self.record_frames = record_frames
        self.clock = pygame.time.Clock()

    def set_render(self, render: bool):
        self.render = render

    def set_framerate(self, framerate):
        self.framerate = framerate

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.render:
            frame = self.env.render()
            if self.record_frames:
                info["rendered_frame"] = frame
        if self.framerate is not None:
            self.clock.tick(self.framerate)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.render:
            self.env.render()

        return obs, info
    
# class VideoRecorder:

#     def __init__(self, frames: List[np.ndarray], fps: int):
#         self.frames = frames
#         self.fps = fps

#     def save_video(self, output_path: str):

#         n_frames, height, width, n_channels = self.frames.shape
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
#         for frame in self.frames:
#             video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#         video.release()

class VideoRecorder:
    def __init__(self, fps: int, dialog_frame_duration_seconds: int):
        self.fps = fps
        self.dialog_frame_duration_seconds = dialog_frame_duration_seconds
        self.video_frames = None

    def parse_frames(self, rendered_frames: List[dict]):
        video_frames = []
        num_repeated_frames_per_dialog = self.dialog_frame_duration_seconds * self.fps
        for frame_info in rendered_frames:
            # Main rendered frame
            video_frames.append(frame_info['gameplay_frame'])
            # Dialog rendered frames, each repeated for dialog_frame_duration_seconds seconds
            for dialog_frame in frame_info['dialog_frames']:
                for _ in range(num_repeated_frames_per_dialog):
                    video_frames.append(dialog_frame)
        self.video_frames = np.array(video_frames)

    def save_video(self, output_path: str):
        if self.video_frames is None:
            raise ValueError("Video frames have not been parsed yet. Call parse_frames() before saving the video.")
        import cv2
        
        n_frames, height, width, n_channels = self.video_frames.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        for frame in self.video_frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()

def is_env_registered(env_id):
    return env_id in registry.keys()

def register_env(env_id="SpaceInvadersEnv", game_type=None, max_episode_steps=10000):
    if is_env_registered(env_id):
        print(f"Environment {env_id} already registered")
        return env_id
    
    if game_type in [GameTypes.COMPETITIVE, GameTypes.COMPETITIVE_TUTORIAL]:
        entry_point = "env.competitive_env:CompetitiveSpaceInvadersEnv"
    elif game_type == GameTypes.COOPERATIVE:
        entry_point = "env.cooperative_env:CooperativeSpaceInvadersEnv"
    else:
        raise ValueError(f"Invalid environment type: {game_type}")
    
    print(f"Registering environment {env_id} with entry point {entry_point} and max episode steps {max_episode_steps}")
    register(
        id=env_id, entry_point=entry_point, max_episode_steps=max_episode_steps
    )
    return env_id

def create_env(
    env_id="SpaceInvadersEnv",
    multiprocessing=False,
    num_envs=1,
    action_space=ActionSpaces.HUMAN_ONLY,
    reward_model=RewardTypes.FULL,
    verbose=False,
    render_mode=RenderModes.DISPLAY_WINDOW.value,
    support_policy=NaoSupportPolicies.EQUAL_SUPPORT,
    rules_based_human_policy=False,
    minimal_complexity_env=False,
    render=False,
    fixed_framerate=None,
    renderTime=True,
    renderFrameNumber=True,
    use_observations=True,
    interactive_mode=InteractiveModes.NONE,
    interactive_players: Optional[List[Players]]=None, # Only used if interactive_mode is not NONE
    num_game_segments=1,
    game_duration_frames=DynamicsConsts.MAX_NUM_FRAMES_PER_GAME,
    independent_victory_score_threshold=None,
    game_type=GameTypes.COMPETITIVE,
    human_weak_player_flag=False,
    shutter_weak_player_flag=False,
    adjust_player_shooting=None,
):    
    if not is_env_registered(env_id):
        raise ValueError(f"Environment {env_id} not registered. Use register_env() to register the environment.")

    if interactive_players is None:
        if interactive_mode == InteractiveModes.NONE:
            interactive_players = []
        else:
            interactive_players = [Players.HUMAN, Players.SHUTTER]

    # Training/eval without live rendering should use the off-screen path to avoid
    # paying for display-window setup.
    if not render and fixed_framerate is None:
        render_mode = RenderModes.RGB_ARRAY.value
    
    if multiprocessing:
        env = make_vec_env(
            env_id=env_id,
            n_envs=num_envs,
            env_kwargs={
                "action_space": action_space,
                "reward_model": reward_model,
                "verbose": verbose,
                "render_mode": render_mode,
                "support_policy": support_policy,
                "rules_based_human_policy": rules_based_human_policy,
                "minimal_complexity_env": minimal_complexity_env,
                "renderTime": renderTime,
                "renderFrameNumber": renderFrameNumber,
                "use_observations": use_observations,
                "interactive_mode": interactive_mode,
                "interactive_players": interactive_players,
                "num_game_segments": num_game_segments,
                "game_duration_frames": game_duration_frames,
                "independent_victory_score_threshold": independent_victory_score_threshold,
                "game_type": game_type,
                "human_weak_player_flag": human_weak_player_flag,
                "shutter_weak_player_flag": shutter_weak_player_flag,
                "adjust_player_shooting": adjust_player_shooting,
            },
        )
        if render or fixed_framerate is not None:
            raise NotImplementedError("Rendering not supported with multiprocessing")

        return env
    else:
        env = gym.make(
            id=env_id,
            action_space=action_space,
            reward_model=reward_model,
            verbose=verbose,
            render_mode=render_mode,
            support_policy=support_policy,
            rules_based_human_policy=rules_based_human_policy,
            minimal_complexity_env=minimal_complexity_env,
            renderTime=renderTime,
            renderFrameNumber=renderFrameNumber,
            use_observations=use_observations,
            interactive_mode=interactive_mode,
            interactive_players=interactive_players,
            num_game_segments=num_game_segments,
            game_duration_frames=game_duration_frames,
            independent_victory_score_threshold=independent_victory_score_threshold,
            game_type=game_type,
            human_weak_player_flag=human_weak_player_flag,
            shutter_weak_player_flag=shutter_weak_player_flag,
            adjust_player_shooting=adjust_player_shooting,
        )
        env = EnvRenderWrapper(env, render=render, framerate=fixed_framerate)
        env.reset()
        return env
    
### End of environment utilities

### Randomness and Reproducibility Utilities
import torch
import random
import numpy as np
def set_seed(seed):
    # NOTE: seeding heavily reduces randomness, but not 100%, so there is still some minor variance in results
    # Set random seed for Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure reproducibility for CUDA
    if torch.cuda.is_available() and seed is not None:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

def set_deterministic_cudnn():
    # Set deterministic behavior for cudnn
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
###

### Readability and beautification utilities
def open_print_section(section_description: str = None, divider_len: int=80):
    print("="* divider_len)
    if section_description is not None:
        print(section_description)
        print("-"* divider_len)

def close_print_section(section_description: str = None, divider_len: int=80):
    if section_description is not None:
        print("-"* divider_len)
        print(section_description)
    print("="* divider_len)
###

### Saving and Loading Utilities

def recursive_remove_key(dictionary, bad_key):
    keys_to_remove = []
    for key, value in dictionary.items():
        if key == bad_key:
            keys_to_remove.append(key)
        elif isinstance(value, dict):
            recursive_remove_key(value, bad_key)
    
    for key in keys_to_remove:
        del dictionary[key]

class SetAndNPEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CustomJSONEncoder(json.JSONEncoder):
    def _enum_to_dict(self, enum_obj: Enum):
        # NOTE: __qualname__ includes nested class names (e.g., OuterClass.InnerClass)
        return {
            "__enum_module__": enum_obj.__class__.__module__,
            "__enum_class__": enum_obj.__class__.__qualname__,
            "name": enum_obj.name,
            "value": enum_obj.value
        }
    
    def _action_to_dict(self, action_obj: Action):
        return {
            "__action_module__": action_obj.__class__.__module__,
            "__action_class__": action_obj.__class__.__name__,
            "dct": action_obj.to_dict()
        }
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray_as_list__": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, Enum):
            return self._enum_to_dict(obj)
        
        if isinstance(obj, Action):
            return self._action_to_dict(obj)
        
        if isinstance(obj, RobotActionRequest):
            return {
                "__robot_action_request_module__": obj.__class__.__module__,
                "__robot_action_request_class__": obj.__class__.__name__,
                "dct": obj.to_dict()
            }

        GameScriptEvent = importlib.import_module("interactive_gameplay.utils").GameScriptEvent
        if isinstance(obj, GameScriptEvent):
            return {
                "__game_script_event_module__": obj.__class__.__module__,
                "__game_script_event_class__": obj.__class__.__name__,
                "dct": obj.to_dict()
            }

        # NOTE: still debugging this functionality
        # if isinstance(obj, dict):
        #     import pdb; pdb.set_trace()
        #     contains_enum_keys = all(isinstance(key, Enum) for key in obj.keys())
        #     if contains_enum_keys:
        #         return {
        #             "__dict_with_enum_keys__": {
        #                 self._enum_to_dict(key): value for key, value in obj.items()
        #             }
        #         }

        return json.JSONEncoder.default(self, obj)
    
class CustomJSONDecoder:

    # @staticmethod
    # def _dict_to_enum(dct):
    #     module_name = dct["__enum_module__"]
    #     class_name = dct["__enum_class__"]
    #     enum_class = getattr(importlib.import_module(module_name), class_name)
    #     return enum_class[dct["name"]]

    @staticmethod
    def _get_class(module_name, class_name):
        """
        Dynamically import a module and retrieve a class from it.
        Args:
            module_name (str): The name of the module to import.
            class_name (str): The name of the class to retrieve. If the class is nested, use dot notation (e.g., 'OuterClass.InnerClass').

        Returns:
            type: The class object.
        """
        module = importlib.import_module(module_name)
        if '.' in class_name:
            # Traverse the class hierarchy to get the final class
            class_hierarchy = class_name.split('.')
            current_class = module
            for cls_name in class_hierarchy:
                current_class = getattr(current_class, cls_name)
            return current_class
        else:
            return getattr(module, class_name)
    
    @staticmethod
    def custom_json_decode(dct):
        if "__ndarray_as_list__" in dct:
            return np.array(dct["__ndarray_as_list__"], dtype=dct["dtype"]).reshape(dct["shape"])
        if "__enum_class__" in dct:
            module_name = dct["__enum_module__"]
            class_name = dct["__enum_class__"]
            #enum_class = getattr(importlib.import_module(module_name), class_name)
            enum_class = CustomJSONDecoder._get_class(module_name, class_name)
            return enum_class[dct["name"]]
            #return CustomJSONDecoder._dict_to_enum(dct)
        if "__action_class__" in dct:
            module_name = dct["__action_module__"]
            class_name = dct["__action_class__"]
            action_class = getattr(importlib.import_module(module_name), class_name)
            return action_class.from_dict(dct["dct"])
        if "__robot_action_request_class__" in dct:
            module_name = dct["__robot_action_request_module__"]
            class_name = dct["__robot_action_request_class__"]
            action_class = getattr(importlib.import_module(module_name), class_name)
            return action_class.from_dict(dct["dct"])
        if "__game_script_event_class__" in dct:
            module_name = dct["__game_script_event_module__"]
            class_name = dct["__game_script_event_class__"]
            event_class = getattr(importlib.import_module(module_name), class_name)
            return event_class.from_dict(dct["dct"])
        
        # NOTE: still debugging this functionality
        # if "__dict_with_enum_keys__" in dct:
        #     import pdb; pdb.set_trace()
        #     return {
        #         CustomJSONDecoder._dict_to_enum(key): value for key, value in dct["__dict_with_enum_keys__"].items()    
        #     }
        return dct

###

### Computational utilities
import torch.nn as nn
import numpy as np

# Function to clone a linear layer
def clone_linear_layer(layer: nn.Linear) -> nn.Linear:
    # Create a new instance of the layer with the same input and output dimensions
    cloned_layer = nn.Linear(layer.in_features, layer.out_features)
    
    # Copy the weights and biases from the original layer to the cloned layer
    cloned_layer.weight.data = layer.weight.data.detach().clone()
    cloned_layer.bias.data = layer.bias.data.detach().clone()
    
    return cloned_layer

import copy
def clone_module_or_layer(module: nn.Module) -> nn.Module:
    """
    Wrapper function to clone either an nn.Linear layer or a generic nn.Module.
    
    Args:
        module (nn.Module): The module to clone.
    
    Returns:
        nn.Module: A cloned version of the input module.
    """
    if isinstance(module, nn.Linear):
        return clone_linear_layer(module)
    else:
        # Use deepcopy for generic nn.Module cloning
        return copy.deepcopy(module)

# Function to round iterable of values marginally to avoid zeros that can crash log functions
def zerosafe_iterable(iterable, nearzero_value=1e-10, inplace=True):
    if inplace:
        for i, x in enumerate(iterable):
            if x == 0:
                iterable[i] = nearzero_value
        return iterable
    else:
        it_type = type(iterable)
        if it_type == np.ndarray:
            return np.where(iterable == 0, nearzero_value, iterable)
        elif it_type == torch.Tensor:
            return torch.where(iterable == 0, nearzero_value, iterable)
        else:
            return it_type(nearzero_value if x == 0 else x for x in iterable)

# Function to calculate the probability density function of a guassian distribution at a particular value x (torch compatible)
def gaussian_pdf(x, mean, std):
    var = std ** 2
    denom = (2 * torch.pi * var) ** 0.5
    num = torch.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom
###
