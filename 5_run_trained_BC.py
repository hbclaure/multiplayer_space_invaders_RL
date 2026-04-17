import argparse
import os
import pprint
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from consts import ActionSpaces, DynamicsConsts, GameTypes, Players, RewardTypes, PlayerShootingAdjustment
from path_consts import ANALYSIS_RESULTS_DIR
from tests import (
    _sanitize_skill_pair,
    evaluate_episodes,
    get_evaluation_threshold,
    save_all_plots,
    save_episode_metrics_csv,
)
from utils import create_env, init_experiment_dir, register_env


DEFAULT_BC_MODEL_DIR = "/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/testing"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained behavior cloning model and generate the same evaluation plots as tests.py.")
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Name of the analysis output directory (default: timestamped directory)",
    )
    parser.add_argument(
        "--manual-model-experiment-name",
        type=str,
        default=None,
        help="BC checkpoint name or folder under testing/ (for example: testing_BC).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of episodes to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--independent-victory-score-threshold",
        type=int,
        default=None,
        help="Optional threshold used by the environment when configured for fairness reward.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the evaluation episodes (default: %(default)s)",
    )
    parser.add_argument(
        "--human-weak-player",
        action="store_true",
        help="Whether the human agent is a weak player (default: %(default)s)",
    )
    parser.add_argument(
        "--shutter-weak-player",
        action="store_true",
        help="Whether the shutter agent is a weak player (default: %(default)s)",
    )
    parser.add_argument(
        "--adjust-player-shooting",
        type=str,
        default="both",
        choices=PlayerShootingAdjustment.values(),
        help="Adjusting the shooting of a player, human, shutter, or both(default: %(default)s)",
    )
    return parser.parse_args()


def build_bc_model(obs_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    )


def resolve_bc_checkpoint(model_name: Optional[str]) -> Tuple[Optional[str], str]:
    candidates = []
    if model_name:
        candidates.extend(
            [
                (os.path.join(DEFAULT_BC_MODEL_DIR, model_name), os.path.join(DEFAULT_BC_MODEL_DIR, model_name, "bc_best_model.pt")),
                (None, os.path.join(DEFAULT_BC_MODEL_DIR, f"{model_name}.pt")),
                (None, os.path.join(DEFAULT_BC_MODEL_DIR, model_name)),
            ]
        )
    candidates.append((DEFAULT_BC_MODEL_DIR, os.path.join(DEFAULT_BC_MODEL_DIR, "bc_best_model.pt")))

    for experiment_dir, path in candidates:
        if os.path.isfile(path):
            return experiment_dir, path

    searched = "\n".join(path for _, path in candidates)
    raise FileNotFoundError(f"Could not find BC checkpoint. Searched:\n{searched}")


class BCPolicyWrapper:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, obs, deterministic: bool = True):
        obs_tensor = torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(obs_tensor)
            action = int(torch.argmax(logits, dim=1).item())
        return action, None


def main():
    args = parse_args()
    print("Running 5_run_trained_BC.py with the following arguments:")
    pprint.pprint(args)

    output_dir = init_experiment_dir(ANALYSIS_RESULTS_DIR, args, wipe_dir=True)
    experiment_dir, checkpoint_path = resolve_bc_checkpoint(args.manual_model_experiment_name)

    env_id = register_env(game_type=GameTypes.COMPETITIVE)
    env = create_env(
        env_id=env_id,
        multiprocessing=False,
        action_space=ActionSpaces.NAO_ONLY,
        reward_model=RewardTypes.NAO_FAIRNESS,
        render=args.render,
        fixed_framerate=DynamicsConsts.FRAMES_PER_SECOND if args.render else None,
        rules_based_human_policy=True,
        independent_victory_score_threshold=args.independent_victory_score_threshold,
        shutter_weak_player_flag=args.shutter_weak_player,
        human_weak_player_flag=args.human_weak_player,
        adjust_player_shooting=args.adjust_player_shooting,
    )

    obs_dim = int(env.observation_space.shape[0])
    model = build_bc_model(obs_dim)
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    wrapped_model = BCPolicyWrapper(model)

    episode_metrics = evaluate_episodes(model=wrapped_model, env=env, num_episodes=args.num_episodes, heuristic_policy_type=None)
    threshold = get_evaluation_threshold(args=args, experiment_dir=experiment_dir or output_dir, env=env)

    csv_path = save_episode_metrics_csv(episode_metrics=episode_metrics, output_dir=output_dir)
    plot_paths = save_all_plots(episode_metrics, output_dir, threshold)

    skill_groups = {}
    for metric in episode_metrics:
        skill_groups.setdefault(metric["skill_pair"], []).append(metric)

    skill_group_outputs = []
    for skill_pair, metrics in sorted(skill_groups.items()):
        skill_output_dir = os.path.join(output_dir, f"skill_pair_{_sanitize_skill_pair(skill_pair)}")
        os.makedirs(skill_output_dir, exist_ok=True)
        skill_csv_path = save_episode_metrics_csv(metrics, skill_output_dir)
        skill_plot_paths = save_all_plots(metrics, skill_output_dir, threshold)
        skill_group_outputs.append((skill_pair, skill_csv_path, skill_plot_paths))

    env.close()

    print(f"Loaded BC checkpoint from {checkpoint_path}")
    print(f"Saved episode metrics to {csv_path}")
    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")
    for skill_pair, skill_csv_path, skill_plot_paths in skill_group_outputs:
        print(f"Saved {skill_pair} episode metrics to {skill_csv_path}")
        for plot_path in skill_plot_paths:
            print(f"Saved {skill_pair} plot to {plot_path}")


if __name__ == "__main__":
    main()
