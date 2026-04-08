import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import torch

from path_consts import ANALYSIS_RESULTS_DIR


DEFAULT_DATASET_ROOT = os.path.join(ANALYSIS_RESULTS_DIR, "behavior_cloning")
ACTION_LABELS = {
    0: "Support Human",
    1: "Support Shutter",
    2: "Center / Idle",
}
SKILL_PAIR_ORDER = ["good/good", "good/bad", "bad/good", "bad/bad"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze collected expert-policy BC datasets.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing behavior cloning datasets grouped by support policy.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="expert_policy_analysis",
        help="Name of the analysis output directory created under analysis_results/.",
    )
    return parser.parse_args()


def find_behavior_cloning_files(dataset_root: str) -> List[Tuple[str, str, str]]:
    datasets = []
    for policy_name in sorted(os.listdir(dataset_root)):
        policy_dir = os.path.join(dataset_root, policy_name)
        if not os.path.isdir(policy_dir):
            continue
        for run_name in sorted(os.listdir(policy_dir)):
            run_dir = os.path.join(policy_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            bc_path = os.path.join(run_dir, "behavior_cloning_data.pt")
            if os.path.exists(bc_path):
                datasets.append((policy_name, run_name, bc_path))
    return datasets


def load_dataset(path: str) -> Dict:
    return torch.load(path, weights_only=False)


def _safe_counter(values) -> Counter:
    return Counter(values if values is not None else [])


def summarize_dataset(policy_name: str, run_name: str, data: Dict) -> Dict[str, object]:
    actions = list(data["actions"])
    episodes = list(data["episode"])
    metadata = data.get("metadata", {})
    human_labels = list(metadata.get("human_skill_label", []))
    shutter_labels = list(metadata.get("shutter_skill_label", []))

    action_counts = _safe_counter(actions)
    unique_episode_ids = sorted(set(episodes))
    skill_pairs = [f"{h}/{s}" for h, s in zip(human_labels, shutter_labels)]
    skill_pair_counts = _safe_counter(skill_pairs)

    summary = {
        "policy_name": policy_name,
        "run_name": run_name,
        "num_samples": len(actions),
        "num_unique_episodes": len(unique_episode_ids),
        "action_0_count": action_counts.get(0, 0),
        "action_1_count": action_counts.get(1, 0),
        "action_2_count": action_counts.get(2, 0),
        "action_0_pct": 100.0 * action_counts.get(0, 0) / max(1, len(actions)),
        "action_1_pct": 100.0 * action_counts.get(1, 0) / max(1, len(actions)),
        "action_2_pct": 100.0 * action_counts.get(2, 0) / max(1, len(actions)),
    }

    for skill_pair in SKILL_PAIR_ORDER:
        summary[f"skill_pair_{skill_pair.replace('/', '_')}_timesteps"] = skill_pair_counts.get(skill_pair, 0)

    episode_skill_pairs = {}
    for episode_id, human_label, shutter_label in zip(episodes, human_labels, shutter_labels):
        episode_skill_pairs.setdefault(episode_id, f"{human_label}/{shutter_label}")
    episode_skill_pair_counts = _safe_counter(episode_skill_pairs.values())
    for skill_pair in SKILL_PAIR_ORDER:
        summary[f"skill_pair_{skill_pair.replace('/', '_')}_episodes"] = episode_skill_pair_counts.get(skill_pair, 0)

    return summary


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_report(path: str, dataset_summaries: List[Dict[str, object]], aggregate_summary: Dict[str, object]):
    with open(path, "w") as f:
        f.write("Expert Policy Dataset Analysis\n")
        f.write("==============================\n\n")

        f.write("Overall Summary\n")
        f.write("---------------\n")
        f.write(f"Total datasets: {aggregate_summary['num_datasets']}\n")
        f.write(f"Total samples: {aggregate_summary['num_samples']}\n")
        f.write(f"Total unique episodes: {aggregate_summary['num_unique_episodes']}\n")
        f.write("Overall action counts:\n")
        for action_idx in sorted(ACTION_LABELS):
            f.write(
                f"  {action_idx} ({ACTION_LABELS[action_idx]}): "
                f"{aggregate_summary[f'action_{action_idx}_count']} "
                f"({aggregate_summary[f'action_{action_idx}_pct']:.2f}%)\n"
            )
        f.write("Overall skill-pair episode counts:\n")
        for skill_pair in SKILL_PAIR_ORDER:
            key = skill_pair.replace("/", "_")
            f.write(f"  {skill_pair}: {aggregate_summary[f'skill_pair_{key}_episodes']}\n")
        f.write("\n")

        f.write("Per-Dataset Summary\n")
        f.write("-------------------\n")
        for summary in dataset_summaries:
            f.write(f"{summary['policy_name']} / {summary['run_name']}\n")
            f.write(f"  samples: {summary['num_samples']}\n")
            f.write(f"  unique episodes: {summary['num_unique_episodes']}\n")
            f.write("  action counts:\n")
            for action_idx in sorted(ACTION_LABELS):
                f.write(
                    f"    {action_idx} ({ACTION_LABELS[action_idx]}): "
                    f"{summary[f'action_{action_idx}_count']} "
                    f"({summary[f'action_{action_idx}_pct']:.2f}%)\n"
                )
            f.write("  skill-pair episode counts:\n")
            for skill_pair in SKILL_PAIR_ORDER:
                key = skill_pair.replace("/", "_")
                f.write(f"    {skill_pair}: {summary[f'skill_pair_{key}_episodes']}\n")
            f.write("\n")


def aggregate_summaries(dataset_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    aggregate = {
        "num_datasets": len(dataset_summaries),
        "num_samples": sum(int(row["num_samples"]) for row in dataset_summaries),
        "num_unique_episodes": sum(int(row["num_unique_episodes"]) for row in dataset_summaries),
    }
    total_samples = max(1, aggregate["num_samples"])

    for action_idx in ACTION_LABELS:
        count_key = f"action_{action_idx}_count"
        aggregate[count_key] = sum(int(row[count_key]) for row in dataset_summaries)
        aggregate[f"action_{action_idx}_pct"] = 100.0 * aggregate[count_key] / total_samples

    for skill_pair in SKILL_PAIR_ORDER:
        key = skill_pair.replace("/", "_")
        timestep_key = f"skill_pair_{key}_timesteps"
        episode_key = f"skill_pair_{key}_episodes"
        aggregate[timestep_key] = sum(int(row[timestep_key]) for row in dataset_summaries)
        aggregate[episode_key] = sum(int(row[episode_key]) for row in dataset_summaries)

    return aggregate


def main():
    args = parse_args()

    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")

    datasets = find_behavior_cloning_files(args.dataset_root)
    if not datasets:
        raise FileNotFoundError(f"No behavior_cloning_data.pt files found under: {args.dataset_root}")

    output_dir = os.path.join(ANALYSIS_RESULTS_DIR, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    dataset_summaries = []
    for policy_name, run_name, bc_path in datasets:
        data = load_dataset(bc_path)
        dataset_summaries.append(summarize_dataset(policy_name, run_name, data))

    aggregate_summary = aggregate_summaries(dataset_summaries)

    summary_csv_path = os.path.join(output_dir, "per_dataset_summary.csv")
    aggregate_json_path = os.path.join(output_dir, "aggregate_summary.json")
    report_path = os.path.join(output_dir, "report.txt")

    write_csv(summary_csv_path, dataset_summaries)
    with open(aggregate_json_path, "w") as f:
        json.dump(aggregate_summary, f, indent=2, sort_keys=True)
    write_text_report(report_path, dataset_summaries, aggregate_summary)

    print(f"Analyzed {len(dataset_summaries)} datasets from {args.dataset_root}")
    print(f"Saved per-dataset summary to {summary_csv_path}")
    print(f"Saved aggregate summary to {aggregate_json_path}")
    print(f"Saved text report to {report_path}")


if __name__ == "__main__":
    main()
