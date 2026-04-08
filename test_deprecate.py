# Standard library imports
import argparse
import os
import pprint
import sys
import time
from typing import Dict, List, TYPE_CHECKING

# Third-party library imports
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from stable_baselines3 import DQN

# Project-specific imports
from agents.policies import SingleAgentPolicy, human_rules_based_policy_from_state
from consts import ActionSpaces, GameTypes, HumanPolicies, NaoSupportPolicies, Players, RewardTypes, DynamicsConsts
from deep_sarsa import DeepSarsa
from models import ValueFunction, detect_welfare_reduction, load_model, rollout_trajectory, get_discounted_future_rewards_at_every_step
from visualization import plot_trajectory, plot_all_welfare_reduction_methods

from path_consts import ANALYSIS_RESULTS_DIR, get_model_path, get_manual_model_path
from utils import (
    close_print_section,
    create_env,
    enable_debug_hook,
    init_experiment_dir,
    open_print_section,
    register_env,
    set_deterministic_cudnn,
    set_seed,
)

# Type hints (conditional imports)
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState
    from models import Trajectory, ValueFunction

# Enable debug hook
enable_debug_hook()

def parse_args():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Model Comparison")
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Name of the experiment (default: %(default)s)",
    )
    parser.add_argument(
        "--exp-description",
        type=str,
        default=None,
        help="Description of the experiment (default: %(default)s)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=RewardTypes.FULL.value,
        choices=RewardTypes.values(),
        help="Which reward function to train with (default: %(default)s)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether or not the games will be rendered (default: %(default)s)",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default=GameTypes.COMPETITIVE.value,
        choices=GameTypes.values(),
        help='Whether or not the game will be "cooperative" or "competitive" (default: %(default)s)',
    )
    parser.add_argument(
        "--human-policy",
        type=str,
        default=HumanPolicies.RULES_BASED.value,
        choices=HumanPolicies.values(),
        help="Human policy to use. (default: %(default)s)",
    )
    parser.add_argument(
        "--joint-agent-action-space",
        action="store_true",
        help="Whether or not to use a joint agent action space (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-support-policy",
        type=str,
        default=NaoSupportPolicies.EQUAL_SUPPORT.value,
        choices=NaoSupportPolicies.values(),
        help="Nao support policy. (default: %(default)s)",
    )
    parser.add_argument(
        "--actual-support-policy",
        type=str,
        default=NaoSupportPolicies.PARTIAL_SUPPORT_HUMAN.value,
        choices=NaoSupportPolicies.values(),
        help="Nao support policy. (default: %(default)s)",
    )
    parser.add_argument(
        "--manual-model-experiment-name",
        type=str,
        default=None,
        help="Manual override for default name of the experiment to use for the expected value model (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--debugging-cut-short",
        action="store_true",
        help="Whether or not to cut the trajectory short for debugging purposes (default: %(default)s)",
    )

    # TODO: deprecate?
    parser.add_argument(
        "--policy-greed",
        type=str,
        default="greedy",
        choices=["epsilon_greedy", "greedy"],
        help="Policy for selecting actions (default: %(default)s)",
    )

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    open_print_section("Running model_comparisons_v2.py with the following arguments:")
    pprint.pprint(args)
    close_print_section()

    # Create the output directory -- prompt to wipe it if it already exists
    output_dir = init_experiment_dir(ANALYSIS_RESULTS_DIR, args, wipe_dir=True)

    # Set seed and deterministic CUDNN for replicability
    set_seed(args.seed)
    set_deterministic_cudnn()

    # Choose the action space
    if args.joint_agent_action_space:
        action_space = ActionSpaces.JOINT_AGENT
    else:
        action_space = ActionSpaces.HUMAN_ONLY

    # Choose whether policy is deterministic
    if args.policy_greed == "greedy":
        deterministic_policy = True
    elif args.policy_greed == "epsilon_greedy":
        deterministic_policy = False
    else:
        raise Exception("Policy must be among ['epsilon_greedy', 'greedy']")
    
    DEBUGGING = args.debugging_cut_short

    # constants
    RETROSPECTIVE_LENGTH_SECONDS = 15
    SHORT_TERM_HORIZON_SECONDS = 5
    if DEBUGGING:
        DEBUGGING_SHORT_HORIZON = False
        if DEBUGGING_SHORT_HORIZON:
            SHORT_TERM_HORIZON_SECONDS = 5
            RETROSPECTIVE_LENGTH_SECONDS = 2.5
        else:
            RETROSPECTIVE_LENGTH_SECONDS = 2.5
            SHORT_TERM_HORIZON_SECONDS = 0.5

    retrospective_length_frames = int(RETROSPECTIVE_LENGTH_SECONDS * DynamicsConsts.FRAMES_PER_SECOND)
    short_term_horizon_frames = int(SHORT_TERM_HORIZON_SECONDS * DynamicsConsts.FRAMES_PER_SECOND)

    if action_space == ActionSpaces.JOINT_AGENT:
        raise NotImplementedError("Joint agent action space not yet implemented")
    elif action_space == ActionSpaces.HUMAN_ONLY:

        if HumanPolicies(args.human_policy) == HumanPolicies.OPTIMAL:
            raise NotImplementedError("Optimal human policy not yet implemented")
        elif HumanPolicies(args.human_policy) == HumanPolicies.RULES_BASED:
            #human_rules_policy_fn = boolean_to_single_agent_action_policy_wrapper(human_rules_based_policy_from_state)
            human_rules_policy = SingleAgentPolicy(Players.HUMAN, human_rules_based_policy_from_state, policy_fn_returns_booleans=True)
            actual_agent_policy = human_rules_policy
            expected_agent_policy = human_rules_policy
        else:
            raise ValueError(f"Invalid human policy type: {HumanPolicies(args.human_policy)}")
    elif action_space == ActionSpaces.SHUTTER_ONLY:
        raise NotImplementedError("Shutter action space not yet implemented")
    elif action_space == ActionSpaces.NAO_ONLY:
        raise NotImplementedError("Nao action space not yet implemented")
    else:
        raise ValueError(f"Invalid action space: {action_space}")

    # Create expected and actual environments
    open_print_section("Creating expected and actual environments:")
    env_id = register_env(game_type=GameTypes(args.game_type))
    env_creation_args = dict(
        env_id=env_id,
        action_space=action_space,
        reward_model=RewardTypes(args.reward_model),
        rules_based_human_policy=HumanPolicies(args.human_policy) == HumanPolicies.RULES_BASED,
        render=args.render,
        fixed_framerate=DynamicsConsts.FRAMES_PER_SECOND if args.render else None
    )
    actual_env_args = {**env_creation_args, "support_policy": NaoSupportPolicies(args.actual_support_policy)}
    expected_env_args = {**env_creation_args, "support_policy": NaoSupportPolicies(args.expected_support_policy)}
    actual_env = create_env(**actual_env_args)
    expected_env = create_env(**expected_env_args)
    close_print_section()

    # Define the human expectations value function
    open_print_section("Loading human expectations value function:")
    if args.manual_model_experiment_name is None:
        _experiment_dir, model_path = get_model_path(
                game_type=GameTypes(args.game_type),
                human_policy=HumanPolicies(args.human_policy),
                nao_support_policy=NaoSupportPolicies(args.expected_support_policy),
                reward_model=RewardTypes(args.reward_model),
                verbose=True
            )
    else:
        _experiment_dir, model_path = get_manual_model_path(args.manual_model_experiment_name)
    print(f"Loading expected value model from {model_path}.")
    
    human_expectations_value_model = load_model(model_path=model_path, seed=args.seed)
    human_expectations_value_function = ValueFunction(
        model=human_expectations_value_model,
        expected_agent_policy=expected_agent_policy
    )
    close_print_section()

    # Roll out the trajectory
    open_print_section("Rolling out trajectory:")
    debug_break_after_reward = False
    if DEBUGGING:
        max_len = 250
        print(f"Debugging mode enabled. Trajectory will be cut short to {max_len} steps.")
        if DEBUGGING_SHORT_HORIZON:
            debug_break_after_reward = True
            print(f"Debugging mode enabled. Trajectory will break after first reward.")
    else:
        max_len = None 
        print(f"Debugging mode disabled. Trajectory will run to completion.")
    trajectory = rollout_trajectory(
        env=actual_env, 
        agent_policy=actual_agent_policy, 
        max_len=max_len,
        debug_break_after_reward=debug_break_after_reward
    )
    trajectory_plot_path = plot_trajectory(
        trajectory=trajectory,
        human_expectations_value_function=human_expectations_value_function,
        output_dir=output_dir,
        filename='trajectory_plot.png'
    )
    print(f"Succesfully rolled out trajectory. Plot saved to {trajectory_plot_path}.")
    close_print_section()

    # Run the methods of detecting welfare reductions
    open_print_section("Detecting welfare reduction:")
    print(f"Using short term horizon of {SHORT_TERM_HORIZON_SECONDS} seconds = {short_term_horizon_frames} frames.")
    print(f"Using retrospective length of {RETROSPECTIVE_LENGTH_SECONDS} seconds = {retrospective_length_frames} frames.")
    if DEBUGGING:
        retrospective_cutoff = 25
        print(f"Debugging mode enabled. Only applying retrospective in the last {retrospective_cutoff} frames.")
    else:
        retrospective_cutoff = None
    reward_values, future_value_values, past_rewards_values, past_rewards_and_future_value_values = detect_welfare_reduction(
        trajectory=trajectory,
        actual_agent_policy=actual_agent_policy,
        expected_agent_policy=expected_agent_policy,
        actual_env=actual_env,
        expected_env=expected_env,
        human_expectations_value_function=human_expectations_value_function,
        retrospective_length=retrospective_length_frames,
        short_term_horizon=short_term_horizon_frames,
        welfare_plot_fn=plot_all_welfare_reduction_methods,
        output_dir=output_dir,
        debugging_retrospective_cutoff=retrospective_cutoff,
    )
    close_print_section()

    # Plot results
    open_print_section("Plotting final results:")
    plot_all_welfare_reduction_methods(
        reward_values=reward_values,
        future_value_values=future_value_values,
        past_rewards_values=past_rewards_values,
        past_rewards_and_future_value_values=past_rewards_and_future_value_values,
        output_dir=output_dir,
    )
    print(f"Successfully plotted final results. Saved to {output_dir}.")

if __name__ == "__main__":
    main()