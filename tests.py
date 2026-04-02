import argparse
import csv
import json
import os
import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from tqdm import tqdm

from agents.support_policies import BaseSupportPolicy
from consts import ActionSpaces, DynamicsConsts, GameTypes, NaoSupportPolicies, Players, RewardTypes, PlayerShootingAdjustment
from deep_sarsa import DeepSarsa
from path_consts import ANALYSIS_RESULTS_DIR, get_manual_model_path
from utils import create_env, init_experiment_dir, register_env


NAO_ACTION_LABELS = {
    0: "Support Human",
    1: "Support Shutter",
    2: "Center / Idle",
}
NUM_TIME_BUCKETS = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model over multiple episodes and save summary plots.")
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
        help="Experiment directory name under experiments/ that contains the trained model.",
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
    parser.add_argument(
        "--nao-controller",
        type=str,
        default="trained",
        choices=["trained", "heuristic"],
        help="Use the trained model or a heuristic Nao support policy (default: %(default)s)",
    )
    parser.add_argument(
        "--support-policy",
        type=str,
        default=NaoSupportPolicies.EQUAL_SUPPORT.value,
        choices=NaoSupportPolicies.values(),
        help="Heuristic Nao support policy to use when --nao-controller heuristic (default: %(default)s)",
    )
    return parser.parse_args()


def load_trained_model(model_path: str, env):
    try:
        return DQN.load(model_path, env=env)
    except Exception:
        return DeepSarsa.load(model_path, env=env)


def get_base_env(env):
    return env.unwrapped


def heuristic_support_policy_to_action(policy, state) -> int:
    policy.update_support(state)
    if policy.support_player == Players.HUMAN:
        return 0
    if policy.support_player == Players.SHUTTER:
        return 1
    if policy.support_player == Players.NAO:
        return 2
    raise ValueError(f"Unsupported heuristic support target: {policy.support_player}")


def get_evaluation_threshold(args, experiment_dir: str, env) -> float:
    if args.independent_victory_score_threshold is not None:
        return float(args.independent_victory_score_threshold)

    args_path = os.path.join(experiment_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            experiment_args = json.load(f)
        saved_threshold = experiment_args.get("independent_victory_score_threshold")
        if saved_threshold is not None:
            return float(saved_threshold)

    return float(env.unwrapped._get_fairness_threshold())


def _finalize_episode_metrics(
    final_state,
    total_frames: int,
    first_half_counts: Dict[Players, int],
    second_half_counts: Dict[Players, int],
    nao_action_counts: Dict[int, int],
    nao_action_bucket_counts: List[Dict[int, int]],
    score_bucket_snapshots: List[Dict[str, float]],
    episode_idx: int,
    human_skill_label: str,
    shutter_skill_label: str,
    human_shooting_threshold: float,
    shutter_shooting_threshold: float,
):
    human_support = final_state.history.support_frame_count[Players.HUMAN]
    shutter_support = final_state.history.support_frame_count[Players.SHUTTER]
    idle_support = max(0, total_frames - human_support - shutter_support)

    human_score_without_nao = final_state.score_state.scores[Players.HUMAN.value]
    shutter_score_without_nao = final_state.score_state.scores[Players.SHUTTER.value]
    nao_to_human = final_state.score_state.scores["NaoForHuman"]
    nao_to_shutter = final_state.score_state.scores["NaoForShutter"]
    human_score_with_nao = human_score_without_nao + nao_to_human
    shutter_score_with_nao = shutter_score_without_nao + nao_to_shutter

    half_frames = max(1, total_frames // 2)
    second_half_total_frames = max(1, total_frames - half_frames)

    metrics = {
        "episode": episode_idx + 1,
        "human_skill_label": human_skill_label,
        "shutter_skill_label": shutter_skill_label,
        "skill_pair": f"{human_skill_label}/{shutter_skill_label}",
        "human_shooting_threshold": human_shooting_threshold,
        "shutter_shooting_threshold": shutter_shooting_threshold,
        "human_support_frames": human_support,
        "shutter_support_frames": shutter_support,
        "idle_support_frames": idle_support,
        "human_score_without_nao": human_score_without_nao,
        "shutter_score_without_nao": shutter_score_without_nao,
        "human_score_with_nao": human_score_with_nao,
        "shutter_score_with_nao": shutter_score_with_nao,
        "nao_to_human": nao_to_human,
        "nao_to_shutter": nao_to_shutter,
        "first_half_human_support_pct": 100.0 * first_half_counts[Players.HUMAN] / half_frames,
        "first_half_shutter_support_pct": 100.0 * first_half_counts[Players.SHUTTER] / half_frames,
        "second_half_human_support_pct": 100.0 * second_half_counts[Players.HUMAN] / second_half_total_frames,
        "second_half_shutter_support_pct": 100.0 * second_half_counts[Players.SHUTTER] / second_half_total_frames,
        "nao_action_support_human_pct": 100.0 * nao_action_counts[0] / max(1, total_frames),
        "nao_action_support_shutter_pct": 100.0 * nao_action_counts[1] / max(1, total_frames),
        "nao_action_center_idle_pct": 100.0 * nao_action_counts[2] / max(1, total_frames),
        "episode_frames": total_frames,
    }
    for bucket_idx, bucket_counts in enumerate(nao_action_bucket_counts):
        bucket_total = max(1, sum(bucket_counts.values()))
        metrics[f"bucket_{bucket_idx + 1}_support_human_pct"] = 100.0 * bucket_counts[0] / bucket_total
        metrics[f"bucket_{bucket_idx + 1}_support_shutter_pct"] = 100.0 * bucket_counts[1] / bucket_total
        metrics[f"bucket_{bucket_idx + 1}_center_idle_pct"] = 100.0 * bucket_counts[2] / bucket_total
        metrics[f"bucket_{bucket_idx + 1}_human_score_with_nao"] = score_bucket_snapshots[bucket_idx]["human_score_with_nao"]
        metrics[f"bucket_{bucket_idx + 1}_shutter_score_with_nao"] = score_bucket_snapshots[bucket_idx]["shutter_score_with_nao"]
    return metrics


def _normalize_skill_label(raw_label: str) -> str:
    if raw_label == "high_skill_player":
        return "good"
    if raw_label == "low_skill_player":
        return "bad"
    return "unknown"


def evaluate_episodes(model, env, num_episodes: int, heuristic_policy=None) -> List[Dict[str, float]]:
    episode_metrics: List[Dict[str, float]] = []
    base_env = get_base_env(env)

    for episode_idx in tqdm(range(num_episodes), desc="Evaluating episodes", unit="episode"):
        obs, info = env.reset()
        config = base_env.config
        human_skill_label = _normalize_skill_label(getattr(base_env, "human_playerSkill_description", "unknown"))
        shutter_skill_label = _normalize_skill_label(getattr(base_env, "shutter_playerSkill_description", "unknown"))
        human_shooting_threshold = float(getattr(base_env, "human_shooting_threshold_this_episode", np.nan))
        shutter_shooting_threshold = float(getattr(base_env, "shutter_shooting_threshold_this_episode", np.nan))
        half_frame = config.game_duration_frames // 2
        first_half_counts = {Players.HUMAN: 0, Players.SHUTTER: 0}
        second_half_counts = {Players.HUMAN: 0, Players.SHUTTER: 0}
        nao_action_counts = {0: 0, 1: 0, 2: 0}
        nao_action_bucket_counts = [{0: 0, 1: 0, 2: 0} for _ in range(NUM_TIME_BUCKETS)]
        score_bucket_snapshots = [
            {"human_score_with_nao": 0.0, "shutter_score_with_nao": 0.0}
            for _ in range(NUM_TIME_BUCKETS)
        ]

        terminated = False
        truncated = False

        while not (terminated or truncated):
            current_frame = base_env.state.time_state.frame
            if heuristic_policy is not None:
                action_idx = heuristic_support_policy_to_action(heuristic_policy, base_env.state)
                action_to_env = action_idx
            else:
                action, _state = model.predict(obs, deterministic=True)
                action_idx = int(action)
                action_to_env = action
            nao_action_counts[action_idx] = nao_action_counts.get(action_idx, 0) + 1
            bucket_idx = min(NUM_TIME_BUCKETS - 1, int((current_frame / max(1, config.game_duration_frames)) * NUM_TIME_BUCKETS))
            nao_action_bucket_counts[bucket_idx][action_idx] += 1

            if action_idx == 0:
                selected_player = Players.HUMAN
            elif action_idx == 1:
                selected_player = Players.SHUTTER
            else:
                selected_player = None

            if selected_player is not None and current_frame < half_frame:
                first_half_counts[selected_player] += 1
            elif selected_player is not None:
                second_half_counts[selected_player] += 1

            obs, reward, terminated, truncated, info = env.step(action_to_env)
            state = base_env.state
            score_bucket_snapshots[bucket_idx] = {
                "human_score_with_nao": state.score_state.scores[Players.HUMAN.value] + state.score_state.scores["NaoForHuman"],
                "shutter_score_with_nao": state.score_state.scores[Players.SHUTTER.value] + state.score_state.scores["NaoForShutter"],
            }

        last_snapshot = {"human_score_with_nao": 0.0, "shutter_score_with_nao": 0.0}
        for bucket_idx, snapshot in enumerate(score_bucket_snapshots):
            if snapshot["human_score_with_nao"] == 0.0 and snapshot["shutter_score_with_nao"] == 0.0 and bucket_idx > 0:
                score_bucket_snapshots[bucket_idx] = last_snapshot.copy()
            else:
                last_snapshot = snapshot.copy()

        final_state = base_env.get_state(copy_state=True)
        episode_metrics.append(
            _finalize_episode_metrics(
                final_state=final_state,
                total_frames=final_state.time_state.frame,
                first_half_counts=first_half_counts,
                second_half_counts=second_half_counts,
                nao_action_counts=nao_action_counts,
                nao_action_bucket_counts=nao_action_bucket_counts,
                score_bucket_snapshots=score_bucket_snapshots,
                episode_idx=episode_idx,
                human_skill_label=human_skill_label,
                shutter_skill_label=shutter_skill_label,
                human_shooting_threshold=human_shooting_threshold,
                shutter_shooting_threshold=shutter_shooting_threshold,
            )
        )

    return episode_metrics


def save_episode_metrics_csv(episode_metrics: List[Dict[str, float]], output_dir: str):
    csv_path = os.path.join(output_dir, "episode_metrics.csv")
    fieldnames = list(episode_metrics[0].keys())
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_metrics)
    return csv_path


def _save_plot(fig, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _sanitize_skill_pair(skill_pair: str) -> str:
    return skill_pair.replace("/", "_")


def plot_average_support_stacked(episode_metrics: List[Dict[str, float]], output_dir: str):
    human_support = np.array([m["human_support_frames"] for m in episode_metrics], dtype=float)
    shutter_support = np.array([m["shutter_support_frames"] for m in episode_metrics], dtype=float)
    idle_support = np.array([m["idle_support_frames"] for m in episode_metrics], dtype=float)

    means = np.array([human_support.mean(), shutter_support.mean(), idle_support.mean()])
    stds = np.array([human_support.std(), shutter_support.std(), idle_support.std()])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Human Support", "Shutter Support", "Center / Idle"], means, yerr=stds, capsize=6, color=["#2f7ed8", "#c42525", "#7cb342"])
    ax.set_ylabel("Support Frames")
    ax.set_title("Average Support Allocation Per Episode")
    return _save_plot(fig, output_dir, "01_average_support_stacked.png")


def plot_two_player_bar(episode_metrics: List[Dict[str, float]], output_dir: str, filename: str, title: str, human_key: str, shutter_key: str, ylabel: str):
    values = np.array([
        [m[human_key] for m in episode_metrics],
        [m[shutter_key] for m in episode_metrics],
    ], dtype=float)
    means = values.mean(axis=1)
    stds = values.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Human", "Shutter"], means, yerr=stds, capsize=6, color=["#2f7ed8", "#c42525"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return _save_plot(fig, output_dir, filename)


def plot_support_by_half(episode_metrics: List[Dict[str, float]], output_dir: str):
    metrics = {
        "First Half / Human": np.array([m["first_half_human_support_pct"] for m in episode_metrics], dtype=float),
        "First Half / Shutter": np.array([m["first_half_shutter_support_pct"] for m in episode_metrics], dtype=float),
        "Second Half / Human": np.array([m["second_half_human_support_pct"] for m in episode_metrics], dtype=float),
        "Second Half / Shutter": np.array([m["second_half_shutter_support_pct"] for m in episode_metrics], dtype=float),
    }

    labels = list(metrics.keys())
    means = np.array([values.mean() for values in metrics.values()])
    stds = np.array([values.std() for values in metrics.values()])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, means, yerr=stds, capsize=6, color=["#4caf50", "#ff9800", "#1e88e5", "#d81b60"])
    ax.set_ylabel("Support Percentage")
    ax.set_ylim(0, 100)
    ax.set_title("Support Percentage by Game Half")
    ax.tick_params(axis="x", rotation=20)
    return _save_plot(fig, output_dir, "06_support_percentage_by_half.png")


def plot_nao_action_distribution(episode_metrics: List[Dict[str, float]], output_dir: str):
    metrics = {
        "Support Human": np.array([m["nao_action_support_human_pct"] for m in episode_metrics], dtype=float),
        "Support Shutter": np.array([m["nao_action_support_shutter_pct"] for m in episode_metrics], dtype=float),
        "Center / Idle": np.array([m["nao_action_center_idle_pct"] for m in episode_metrics], dtype=float),
    }

    labels = list(metrics.keys())
    means = np.array([values.mean() for values in metrics.values()])
    stds = np.array([values.std() for values in metrics.values()])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=stds, capsize=6, color=["#2f7ed8", "#c42525", "#7cb342"])
    ax.set_ylabel("Action Percentage")
    ax.set_ylim(0, 100)
    ax.set_title("Average Nao Action Distribution")
    return _save_plot(fig, output_dir, "07_nao_action_distribution.png")


def plot_nao_action_distribution_over_time(episode_metrics: List[Dict[str, float]], output_dir: str):
    bucket_labels = [f"T{i}" for i in range(1, NUM_TIME_BUCKETS + 1)]
    x = np.arange(1, NUM_TIME_BUCKETS + 1)
    series = {
        "Support Human": {
            "color": "#2f7ed8",
            "values": np.array(
                [[m[f"bucket_{bucket_idx}_support_human_pct"] for bucket_idx in range(1, NUM_TIME_BUCKETS + 1)] for m in episode_metrics],
                dtype=float,
            ),
        },
        "Support Shutter": {
            "color": "#c42525",
            "values": np.array(
                [[m[f"bucket_{bucket_idx}_support_shutter_pct"] for bucket_idx in range(1, NUM_TIME_BUCKETS + 1)] for m in episode_metrics],
                dtype=float,
            ),
        },
        "Center / Idle": {
            "color": "#7cb342",
            "values": np.array(
                [[m[f"bucket_{bucket_idx}_center_idle_pct"] for bucket_idx in range(1, NUM_TIME_BUCKETS + 1)] for m in episode_metrics],
                dtype=float,
            ),
        },
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, payload in series.items():
        means = payload["values"].mean(axis=0)
        stds = payload["values"].std(axis=0)
        color = payload["color"]
        ax.plot(x, means, label=label, color=color, linewidth=2)
        ax.fill_between(x, np.clip(means - stds, 0, 100), np.clip(means + stds, 0, 100), color=color, alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Action Percentage")
    ax.set_xlabel("Game Time")
    ax.set_ylim(0, 100)
    ax.set_title("Nao Action Distribution Over Time")
    ax.legend()
    return _save_plot(fig, output_dir, "08_nao_action_distribution_over_time.png")


def plot_scores_over_time(episode_metrics: List[Dict[str, float]], output_dir: str):
    bucket_labels = [f"T{i}" for i in range(1, NUM_TIME_BUCKETS + 1)]
    x = np.arange(1, NUM_TIME_BUCKETS + 1)
    series = {
        "Human Score": {
            "color": "#2f7ed8",
            "values": np.array(
                [[m[f"bucket_{bucket_idx}_human_score_with_nao"] for bucket_idx in range(1, NUM_TIME_BUCKETS + 1)] for m in episode_metrics],
                dtype=float,
            ),
        },
        "Shutter Score": {
            "color": "#c42525",
            "values": np.array(
                [[m[f"bucket_{bucket_idx}_shutter_score_with_nao"] for bucket_idx in range(1, NUM_TIME_BUCKETS + 1)] for m in episode_metrics],
                dtype=float,
            ),
        },
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, payload in series.items():
        means = payload["values"].mean(axis=0)
        stds = payload["values"].std(axis=0)
        color = payload["color"]
        ax.plot(x, means, label=label, color=color, linewidth=2)
        ax.fill_between(x, np.maximum(means - stds, 0), means + stds, color=color, alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Score With Nao Support")
    ax.set_xlabel("Game Time")
    ax.set_title("Scores Over Time")
    ax.legend()
    return _save_plot(fig, output_dir, "09_scores_over_time.png")


def plot_threshold_attainment(episode_metrics: List[Dict[str, float]], output_dir: str, threshold: float):
    human_reached = np.array([m["human_score_with_nao"] >= threshold for m in episode_metrics], dtype=float)
    shutter_reached = np.array([m["shutter_score_with_nao"] >= threshold for m in episode_metrics], dtype=float)
    both_reached = np.array(
        [
            (m["human_score_with_nao"] >= threshold) and (m["shutter_score_with_nao"] >= threshold)
            for m in episode_metrics
        ],
        dtype=float,
    )

    means = 100.0 * np.array([human_reached.mean(), shutter_reached.mean(), both_reached.mean()])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Human", "Shutter", "Both"], means, color=["#2f7ed8", "#c42525", "#7cb342"])
    ax.set_ylabel("Episodes Reaching Threshold (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Threshold Attainment Rate (Threshold = {int(threshold)})")
    return _save_plot(fig, output_dir, "10_threshold_attainment.png")


def plot_skill_pair_distribution(episode_metrics: List[Dict[str, float]], output_dir: str):
    ordered_pairs = ["good/good", "good/bad", "bad/good", "bad/bad"]
    counts = np.array([sum(1 for m in episode_metrics if m["skill_pair"] == pair) for pair in ordered_pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered_pairs, counts, color=["#43a047", "#fb8c00", "#1e88e5", "#e53935"])
    ax.set_ylabel("Episodes")
    ax.set_title("Episode Skill-Pair Distribution")
    ax.tick_params(axis="x", rotation=15)
    return _save_plot(fig, output_dir, "11_skill_pair_distribution.png")


def plot_threshold_confusion_matrix(episode_metrics: List[Dict[str, float]], output_dir: str, threshold: float):
    human_labels = ["good", "bad"]
    shutter_labels = ["good", "bad"]
    values = np.full((len(shutter_labels), len(human_labels)), np.nan, dtype=float)
    counts = np.zeros((len(shutter_labels), len(human_labels)), dtype=int)

    for row_idx, shutter_label in enumerate(shutter_labels):
        for col_idx, human_label in enumerate(human_labels):
            matching = [
                m for m in episode_metrics
                if m["human_skill_label"] == human_label and m["shutter_skill_label"] == shutter_label
            ]
            counts[row_idx, col_idx] = len(matching)
            if matching:
                both_reached_rate = 100.0 * np.mean([
                    (m["human_score_with_nao"] >= threshold) and (m["shutter_score_with_nao"] >= threshold)
                    for m in matching
                ])
                values[row_idx, col_idx] = both_reached_rate

    masked_values = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(masked_values, cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(human_labels)))
    ax.set_xticklabels(human_labels)
    ax.set_yticks(np.arange(len(shutter_labels)))
    ax.set_yticklabels(shutter_labels)
    ax.set_xlabel("Human Skill Type")
    ax.set_ylabel("Shutter Skill Type")
    ax.set_title("Both Players Reaching Threshold by Skill Pair (%)")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            cell_value = values[row_idx, col_idx]
            label = f"N/A\nn={counts[row_idx, col_idx]}" if np.isnan(cell_value) else f"{cell_value:.1f}%\nn={counts[row_idx, col_idx]}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Both Reach Threshold (%)")
    return _save_plot(fig, output_dir, "12_threshold_confusion_matrix.png")


def plot_shooting_threshold_distribution(episode_metrics: List[Dict[str, float]], output_dir: str):
    human_thresholds = np.array([m["human_shooting_threshold"] for m in episode_metrics], dtype=float)
    shutter_thresholds = np.array([m["shutter_shooting_threshold"] for m in episode_metrics], dtype=float)
    human_thresholds = human_thresholds[~np.isnan(human_thresholds)]
    shutter_thresholds = shutter_thresholds[~np.isnan(shutter_thresholds)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 20
    if human_thresholds.size:
        ax.hist(human_thresholds, bins=bins, alpha=0.5, color="#2f7ed8", label="Human", density=False)
    if shutter_thresholds.size:
        ax.hist(shutter_thresholds, bins=bins, alpha=0.5, color="#c42525", label="Shutter", density=False)
    ax.set_xlabel("Shooting Threshold")
    ax.set_ylabel("Episode Count")
    ax.set_title("Human vs Shutter Shooting Threshold Distribution")
    ax.legend()
    return _save_plot(fig, output_dir, "13_shooting_threshold_distribution.png")


def save_all_plots(episode_metrics: List[Dict[str, float]], output_dir: str, threshold: float):
    return [
        plot_average_support_stacked(episode_metrics, output_dir),
        plot_two_player_bar(
            episode_metrics,
            output_dir,
            "02_scores_without_nao_support.png",
            "Scores Without Nao Support",
            "human_score_without_nao",
            "shutter_score_without_nao",
            "Score",
        ),
        plot_two_player_bar(
            episode_metrics,
            output_dir,
            "03_scores_with_nao_support.png",
            "Scores With Nao Support",
            "human_score_with_nao",
            "shutter_score_with_nao",
            "Score",
        ),
        plot_two_player_bar(
            episode_metrics,
            output_dir,
            "04_nao_contributions.png",
            "Nao Score Contribution",
            "nao_to_human",
            "nao_to_shutter",
            "Score",
        ),
        plot_two_player_bar(
            episode_metrics,
            output_dir,
            "05_support_time.png",
            "Support Time Per Episode",
            "human_support_frames",
            "shutter_support_frames",
            "Frames",
        ),
        plot_support_by_half(episode_metrics, output_dir),
        plot_nao_action_distribution(episode_metrics, output_dir),
        plot_nao_action_distribution_over_time(episode_metrics, output_dir),
        plot_scores_over_time(episode_metrics, output_dir),
        plot_threshold_attainment(episode_metrics, output_dir, threshold),
        plot_skill_pair_distribution(episode_metrics, output_dir),
        plot_threshold_confusion_matrix(episode_metrics, output_dir, threshold),
        plot_shooting_threshold_distribution(episode_metrics, output_dir),
    ]


def main():
    args = parse_args()
    print("Running tests.py with the following arguments:")
    pprint.pprint(args)

    if args.nao_controller == "trained" and not args.manual_model_experiment_name:
        raise ValueError("--manual-model-experiment-name is required when --nao-controller trained")

    output_dir = init_experiment_dir(ANALYSIS_RESULTS_DIR, args, wipe_dir=True)
    experiment_dir = None
    model_path = None
    if args.manual_model_experiment_name:
        experiment_dir, model_path = get_manual_model_path(args.manual_model_experiment_name)

    env_id = register_env(game_type=GameTypes.COMPETITIVE)
    env = create_env(
        env_id=env_id,
        multiprocessing=False,
        action_space=ActionSpaces.NAO_ONLY,
        reward_model=RewardTypes.NAO_FAIRNESS,
        render=args.render,
        fixed_framerate=DynamicsConsts.FRAMES_PER_SECOND if args.render else None,
        rules_based_human_policy=True,
        support_policy=NaoSupportPolicies(args.support_policy),
        independent_victory_score_threshold=args.independent_victory_score_threshold,
        shutter_weak_player_flag = args.shutter_weak_player ,
        human_weak_player_flag = args.human_weak_player,
        adjust_player_shooting = args.adjust_player_shooting,

        
    )

    model = None
    heuristic_policy = None
    if args.nao_controller == "trained":
        model = load_trained_model(model_path=model_path, env=env)
    else:
        heuristic_policy = BaseSupportPolicy.instantiate_support_policy(NaoSupportPolicies(args.support_policy))

    episode_metrics = evaluate_episodes(model=model, env=env, num_episodes=args.num_episodes, heuristic_policy=heuristic_policy)
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

    print(f"Saved episode metrics to {csv_path}")
    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")
    for skill_pair, skill_csv_path, skill_plot_paths in skill_group_outputs:
        print(f"Saved {skill_pair} episode metrics to {skill_csv_path}")
        for plot_path in skill_plot_paths:
            print(f"Saved {skill_pair} plot to {plot_path}")


if __name__ == "__main__":
    main()
