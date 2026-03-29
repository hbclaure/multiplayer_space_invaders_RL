import sys
import os
import time
import argparse
import pprint
import copy
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.envs.registration import register

from deep_sarsa import DeepSarsa

from path_consts import EXPERIMENTS_DIR
from utils import init_experiment_dir, create_env, register_env
from env_utils import boolean_policy_to_index
from consts import DynamicsConsts, ObservationConsts, RewardTypes, GameTypes, RenderModes, HumanPolicies, NaoSupportPolicies, ActionSpaces, MinimalComplexityHumanPolicies, Players
from agents.policies import human_rules_based_policy_from_obs, create_exploration_policy_fn
#by running: python train_new.py --timesteps 100 --exp-name baseline_run_2 --support-policy biasedSupportLeft you are training the human 

# TO DO: Need to add render mode as an adjustable parameter when you want to save frames (rgb_array)

# TODO: bundle arguments into a function importable, and add a function to fill in missing arguments with defaults
    # currently, any loading/analysis code that uses new arguments will fail when those arguments aren't present in older experiments
#  currently usable support policies:

# humanOnly
# shutterOnly
# equalizeScores
# equalizeHistory
# equalizeHistoryEven
# idleSupport
# equalizeScoresNeutral
# biasedSupportLeft
# biasedSupportRight
def linear_schedule(initial_value: float):
    """
    Define a linear learning rate scheduler
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def exponential_schedule(initial_value, fractional_decrease=250):
    """
    Define an exponentially decaying learning rate schedule
    """

    def func(progress_remaining: float) -> float:
        return np.exp(-np.log(fractional_decrease) * (1 - progress_remaining)) * initial_value

    return func


def run_trained_model(model, env):
    """
    Render model that was trained
    """
    print(f"Rendering the environment at {env.framerate} frames per second.")

    obs, info = env.reset()
    for episode in range(10):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            action, _states = model.predict(obs, deterministic=True)
            #print(f"Action taken: {action}")  # Debugging print statement

            obs, rewards, terminated, truncated, info = env.step(action)
            # print(f"Episode: {episode}, Reward: {rewards}")


from consts import EnhancedEnum
class LR_Schedule(EnhancedEnum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    NONE = "none"


class RewardComponentsTensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        component_values = {}

        for info in infos:
            reward_components = info.get("reward_components", {})
            for name, value in reward_components.items():
                component_values.setdefault(name, []).append(float(value))

        for name, values in component_values.items():
            if values:
                self.logger.record(f"reward_components/{name}", float(np.mean(values)))

        return True

def parse_args():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning Model Training")
    parser.add_argument(
        "--breakpoint-on-exception",
        action="store_true",
        help="Whether or not to automatically drop a breakpoint upon exception (default: %(default)s)",
    )
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
        "--legacy-file-hierarchy",
        action="store_true",
        help="Whether or not to use the legacy file hierarchy (default: %(default)s)",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Whether or not to use multiprocessing (default: %(default)s)",
    )
    parser.add_argument(
        "--num-cpu",
        type=int,
        default=1,
        help="Number of CPUs for multiprocessing (default: %(default)s)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Number of timesteps for training (default: %(default)s)",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.25,
        help="Fraction of training period until reaching final exploration epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for training (default: %(default)s)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=LR_Schedule.NONE.value,
        choices=LR_Schedule.values(),
        help="Learning rate schedule for training (default: %(default)s)",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=1.0,
        help="Discount factor for rewards (default: %(default)s)",
    )
    parser.add_argument(
        "--independent-victory-score-threshold",
        type=int,
        default=None,
        help="Score threshold used for per-player victory checks and fairness reward shaping (default: %(default)s)",
    )
    parser.add_argument(
        "--disadvantaged-player",
        type=str,
        default=None,
        choices=[Players.HUMAN.value, Players.SHUTTER.value, "both"],
        help="Optional scripted player to handicap during training/evaluation renders. (default: %(default)s)",
    )
    parser.add_argument(
        "--disadvantaged-idle-prob",
        type=float,
        default=0.0,
        help="Probability that the disadvantaged scripted player does nothing on a step. (default: %(default)s)",
    )
    parser.add_argument(
        "--disadvantaged-extra-shot-cooldown-frames",
        type=int,
        default=0,
        help="Extra shooting cooldown, in frames, applied to the disadvantaged scripted player. (default: %(default)s)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=RewardTypes.FULL.value,
        choices=RewardTypes.values(),
        help="Which reward function to train with (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether or not info will be passed at each step (default: %(default)s)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether or not the game will be rendered after training (default: %(default)s)",
    )
    parser.add_argument(
        "--render-during-training",
        action="store_true",
        help="Whether or not to render the game during training (default: %(default)s)",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default=GameTypes.COMPETITIVE.value,
        choices=GameTypes.values(),
        help='Whether or not the game will be "cooperative" or "competitive" (default: %(default)s)',
    )
    parser.add_argument(
        "--support-policy",
        type=str,
        default=NaoSupportPolicies.EQUAL_SUPPORT.value,
        choices=NaoSupportPolicies.values(),
        help="Nao support policy. (default: %(default)s)",
    )
    parser.add_argument(
        "--human-policy",
        type=str,
        default=HumanPolicies.OPTIMAL.value,
        choices=HumanPolicies.values(),
        help="Human policy to use. (default: %(default)s)",
    )
    parser.add_argument(
        "--joint-agent-action-space",
        action="store_true",
        help="Whether or not to use a joint agent action space (default: %(default)s)",
    )
    parser.add_argument(
        "--minimal-complexity-env",
        action="store_true",
        help="Whether or not to use a minimal complexity environment (default: %(default)s)",
    )
    parser.add_argument(
        "--minimal-complexity-human-policy",
        type=str,
        default=None,
        choices=MinimalComplexityHumanPolicies.values(),
        help="Minimal complexity human policy to use. (default: %(default)s)",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Running train.py with the following arguments:")
    pprint.pprint(args)

    def validate_args(args):
        # Note: not really necessary with choices in argparse, but just in case
        if not RewardTypes.contains_value(args.reward_model):
            raise ValueError(f"Invalid reward model: {args.reward_model}")
        if not GameTypes.contains_value(args.game_type):
            raise ValueError(f"Invalid game type: {args.game_type}")
        if not NaoSupportPolicies.contains_value(args.support_policy):
            raise ValueError(f"Invalid support policy: {args.support_policy}")
        if not HumanPolicies.contains_value(args.human_policy):
            raise ValueError(f"Invalid human policy: {args.human_policy}")

        if args.multiprocessing and args.num_cpu <= 1:
            warning_msg = "Multiprocessing is enabled but the number of CPUs is less than or equal to 1. Increase --num-cpu to take advantage of multiprocessing."
            user_input = input(f"{warning_msg} Do you want to continue? (y/n): ")
            if user_input.lower() != "y":
                sys.exit(0)

        if args.minimal_complexity_env:
            if args.reward_model != RewardTypes.MINIMAL_COMPLEXITY.value:
                raise ValueError(f"Minimal complexity environment requires minimal complexity reward model. Use --reward-model {RewardTypes.MINIMAL_COMPLEXITY.value}")
            if args.joint_agent_action_space:
                raise ValueError("Minimal complexity environment does not support joint agent action space. Omit --joint-agent-action-space")
            if args.human_policy != HumanPolicies.RULES_BASED.value:
                raise ValueError(f"Minimal complexity environment requires rules-based human policy. Use --human-policy {HumanPolicies.RULES_BASED.value}")
            if args.minimal_complexity_human_policy is None:
                raise ValueError("Minimal complexity environment requires minimal complexity human policy. Use --minimal-complexity-human-policy")
        if not (0.0 <= args.disadvantaged_idle_prob <= 1.0):
            raise ValueError("--disadvantaged-idle-prob must be between 0 and 1")
        if args.disadvantaged_extra_shot_cooldown_frames < 0:
            raise ValueError("--disadvantaged-extra-shot-cooldown-frames must be non-negative")

    validate_args(args)

    # Constants for training (don't foresee changing these values, so not commandline arguments)
    EXPLORATION_INITIAL_EPS = 1  # Start with full exploration
    EXPLORATION_FINAL_EPS = 0.0 # end with no exploration to fit exact potentially non-optimal policies of agents
    RENDER_MODE = RenderModes.DISPLAY_WINDOW.value

    # Informative printout of the training setup
    print("Fixed Constants:")
    print(f"  Exploration initial epsilon: {EXPLORATION_INITIAL_EPS}")
    print(f"  Exploration final epsilon: {EXPLORATION_FINAL_EPS}")
    print(f"  Render mode: {RENDER_MODE}")


    # If applicable, set the breakpoint on exception
    if args.breakpoint_on_exception:
        from utils import enable_debug_hook
        enable_debug_hook()
    action_space = ActionSpaces.NAO_ONLY

    # if args.joint_agent_action_space:
    #     action_space = ActionSpaces.JOINT_AGENT
    # else:
    #     action_space = ActionSpaces.HUMAN_ONLY

    if args.minimal_complexity_env:
        print("Training with minimal complexity environment: simplified observations, only human action space, rules-based human policy, and minimal complexity reward model. Nao support policy will be ignored")
        print("Hacky overwriting DynamicsConsts.MAX_NUM_FRAMES_PER_GAME to shorten horizon")
        from units_utils import seconds_to_frames
        DynamicsConsts.MAX_NUM_FRAMES_PER_GAME = seconds_to_frames(num_seconds=2, frames_per_second=DynamicsConsts.FRAMES_PER_SECOND)

    env_id = register_env(game_type=GameTypes(args.game_type))
    env_creation_args = dict(
        env_id=env_id,
        multiprocessing=args.multiprocessing,
        num_envs=args.num_cpu,
        action_space=action_space,
        reward_model=RewardTypes(args.reward_model),
        verbose=args.verbose,
        render_mode=RENDER_MODE,
        support_policy=NaoSupportPolicies(args.support_policy),
        rules_based_human_policy=HumanPolicies(args.human_policy) == HumanPolicies.RULES_BASED,
        minimal_complexity_env=args.minimal_complexity_env,
        independent_victory_score_threshold=args.independent_victory_score_threshold,
        disadvantaged_player=args.disadvantaged_player,
        disadvantaged_idle_prob=args.disadvantaged_idle_prob,
        disadvantaged_extra_shot_cooldown_frames=args.disadvantaged_extra_shot_cooldown_frames,
        render=args.render_during_training,
        fixed_framerate=None
    )
    env = create_env(**env_creation_args)

    # point to folder to store run information
    if args.legacy_file_hierarchy:
        log_dir = "./data/tensorboard_logs"
        model_dir = "./data/models/"  # point to folder to store model information
        model_filename = f"MAC10M_{args.lr}_{args.exploration_fraction}_{EXPLORATION_FINAL_EPS}_{args.game_type}"
        tb_log_name = f"{args.reward_model}_{args.timesteps/1000000}m_clipped_lr{args.lr}_[64,64]_{args.game_type}_{args.support_policy}"

    else:
        exp_dir = init_experiment_dir(EXPERIMENTS_DIR, args, wipe_dir=True)
        log_dir = os.path.join(exp_dir, "tensorboard_logs")
        model_dir = os.path.join(exp_dir, "models")
        
        tb_log_name = f"env={args.game_type}__support={args.support_policy}__reward={args.reward_model}"
        model_filename = f"{tb_log_name}.pt"

    # TODO: test this
    if HumanPolicies(args.human_policy) == HumanPolicies.OPTIMAL:
        exploration_policy_fn = None
    elif HumanPolicies(args.human_policy) == HumanPolicies.RULES_BASED:
        if args.minimal_complexity_env:
            if MinimalComplexityHumanPolicies(args.minimal_complexity_human_policy) == MinimalComplexityHumanPolicies.SHOOT_WHEN_READY:
                from agents.policies import minimal_human_rules_based_policy_from_obs
                policy_from_obs_fn = minimal_human_rules_based_policy_from_obs
            elif MinimalComplexityHumanPolicies(args.minimal_complexity_human_policy) == MinimalComplexityHumanPolicies.ALWAYS_SHOOT:
                from agents.policies import minimal_always_shoot_human_rules_based_policy_from_obs
                policy_from_obs_fn = minimal_always_shoot_human_rules_based_policy_from_obs
            else:
                raise ValueError(f"Invalid minimal complexity human policy: {args.minimal_complexity_human_policy}")
        else:
            policy_from_obs_fn = human_rules_based_policy_from_obs
        exploration_policy_fn = create_exploration_policy_fn(
            policy_from_obs_fn=policy_from_obs_fn,
            num_stacked_observations=ObservationConsts.RECENT_STATES_BUFFER_SIZE,
        )
    else:
        raise ValueError(f"Invalid human policy: {args.human_policy}")

    # Custom network architecture
    policy_kwargs = dict(net_arch=[64, 64], activation_fn=torch.nn.ReLU)  # 128, 128

    # model
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    if LR_Schedule(args.lr_schedule) == LR_Schedule.LINEAR:
        learning_rate = linear_schedule(args.lr)
    elif LR_Schedule(args.lr_schedule) == LR_Schedule.EXPONENTIAL:
        learning_rate = exponential_schedule(args.lr)
    else:
        learning_rate = args.lr
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        gamma=args.discount_factor,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        tensorboard_log=log_dir,
        device=device,
    )
    # model = DeepSarsa(
    #     "MlpPolicy",
    #     env,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,  # what information to show at terminal
    #     exploration_initial_eps=EXPLORATION_INITIAL_EPS,
    #     exploration_fraction=args.exploration_fraction,
    #     exploration_final_eps=EXPLORATION_FINAL_EPS,
    #     tensorboard_log=log_dir,
    #     learning_rate=learning_rate,
    #     learning_starts=0,
    #     gamma=args.discount_factor,
    #     device=device,
    #     exploration_policy_fn=exploration_policy_fn,
    #     train_freq=(1, "episode"),
    #     gradient_steps=-1,
    #     target_update_interval=1,#DynamicsConsts.MAX_NUM_FRAMES_PER_GAME,
    # )

    # Training the model
    print(f"Training the model for {args.timesteps} timesteps...")
    print(
        f"Exploration rate: starting at {EXPLORATION_INITIAL_EPS}, "
        f"decaying toward {EXPLORATION_FINAL_EPS} over "
        f"{args.exploration_fraction * 100}% of training"
    )
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=4,
        callback=RewardComponentsTensorboardCallback(),
        tb_log_name=tb_log_name,
        progress_bar=True
    )
    end_time = time.time()
    model.save(os.path.join(model_dir, model_filename))

    # Calculate and print the time taken to train
    training_time = end_time - start_time
    print(f"Time taken to train the model: {training_time:.2f} seconds")


    # If you want to render trained model.
    if args.render:

        eval_env_creation_args = copy.deepcopy(env_creation_args)
        eval_env_creation_args["multiprocessing"] = False
        eval_env_creation_args["render"] = True
        eval_env_creation_args["fixed_framerate"] = DynamicsConsts.FRAMES_PER_SECOND

        eval_env = create_env(**eval_env_creation_args)

        print("Rendering the trained model...")
        run_trained_model(model, eval_env)

if __name__ == "__main__":
    main()
