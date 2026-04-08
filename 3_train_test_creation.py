import torch
import os
from sklearn.model_selection import train_test_split
from collections import Counter



path_to_bc_folder = "/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/analysis_results/behavior_cloning"
output_dir = "/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/analysis_results/behavior_cloning_splits"


def summarize_split(name, data):
    lengths = {
        "episode": len(data["episode"]),
        "frame_number": len(data["frame_number"]),
        "observations": len(data["observations"]),
        "actions": len(data["actions"]),
        "dones": len(data["dones"]),
        "human_skill_label": len(data["metadata"]["human_skill_label"]),
        "shutter_skill_label": len(data["metadata"]["shutter_skill_label"]),
        "human_shooting_threshold": len(data["metadata"]["human_shooting_threshold"]),
        "shutter_shooting_threshold": len(data["metadata"]["shutter_shooting_threshold"]),
        "expert_policy": len(data["metadata"]["expert_policy"]),
    }
    all_same_length = len(set(lengths.values())) == 1

    action_counts = Counter(data["actions"])
    expert_policy_counts = Counter(data["metadata"]["expert_policy"])
    skill_pair_counts = Counter(
        f"{human}/{shutter}"
        for human, shutter in zip(
            data["metadata"]["human_skill_label"],
            data["metadata"]["shutter_skill_label"],
        )
    )

    print(f"\n{name} split")
    print("-" * (len(name) + 6))
    print(f"same lengths across fields: {all_same_length}")
    for key, value in lengths.items():
        print(f"  {key}: {value}")
    print("action distribution:")
    for action in [0, 1, 2]:
        count = action_counts.get(action, 0)
        pct = 100.0 * count / max(1, len(data["actions"]))
        print(f"  {action}: {count} ({pct:.2f}%)")
    print("expert-policy distribution:")
    for policy, count in sorted(expert_policy_counts.items()):
        pct = 100.0 * count / max(1, len(data["actions"]))
        print(f"  {policy}: {count} ({pct:.2f}%)")
    print("skill-pair distribution:")
    for skill_pair in ["good/good", "good/bad", "bad/good", "bad/bad"]:
        count = skill_pair_counts.get(skill_pair, 0)
        pct = 100.0 * count / max(1, len(data["actions"]))
        print(f"  {skill_pair}: {count} ({pct:.2f}%)")

combined_train = {
    "episode": [],
    "frame_number": [],
    "observations": [],
    "actions": [],
    "dones": [],
    "metadata": {
        "human_skill_label": [],
        "shutter_skill_label": [],
        "human_shooting_threshold": [],
        "shutter_shooting_threshold": [],
        "expert_policy": [],
    },
}

combined_test = {
    "episode": [],
    "frame_number": [],
    "observations": [],
    "actions": [],
    "dones": [],
    "metadata": {
        "human_skill_label": [],
        "shutter_skill_label": [],
        "human_shooting_threshold": [],
        "shutter_shooting_threshold": [],
        "expert_policy": [],
    },
}

combined_eval = {
    "episode": [],
    "frame_number": [],
    "observations": [],
    "actions": [],
    "dones": [],
    "metadata": {
        "human_skill_label": [],
        "shutter_skill_label": [],
        "human_shooting_threshold": [],
        "shutter_shooting_threshold": [],
        "expert_policy": [],
    },
}
#get pt_dataset['episode']
for expert_policy in os.listdir(path_to_bc_folder):
    data_folder = os.path.join(path_to_bc_folder, expert_policy, "bc_mix")
    for dataset in os.listdir(data_folder):
        if dataset == "behavior_cloning_data.pt":
            pt_dataset = torch.load(os.path.join(data_folder, dataset), weights_only=False)
            #pt_dataset= torch.load(dataset)
            train_idx, test_idx = train_test_split(
                sorted(set(pt_dataset['episode'])),
                test_size=0.2,
                random_state=42
                )
            
            train_idx, eval_idx = train_test_split(
                train_idx,
                test_size=0.1,
                random_state=42
                )
            
            train_indices = [i for i, ep in enumerate(pt_dataset['episode']) if ep in train_idx]
            test_indices = [i for i, ep in enumerate(pt_dataset['episode']) if ep in test_idx]
            eval_indices = [i for i, ep in enumerate(pt_dataset['episode']) if ep in eval_idx]
            train_data = {
                "episode": [pt_dataset["episode"][i] for i in train_indices],
                "frame_number": [pt_dataset["frame_number"][i] for i in train_indices],
                "observations": [pt_dataset["observations"][i] for i in train_indices],
                "actions": [pt_dataset["actions"][i] for i in train_indices],
                "dones": [pt_dataset["dones"][i] for i in train_indices],
                "metadata": {
                    "human_skill_label": [pt_dataset["metadata"]["human_skill_label"][i] for i in train_indices],
                    "shutter_skill_label": [pt_dataset["metadata"]["shutter_skill_label"][i] for i in train_indices],
                    "human_shooting_threshold": [pt_dataset["metadata"]["human_shooting_threshold"][i] for i in train_indices],
                    "shutter_shooting_threshold": [pt_dataset["metadata"]["shutter_shooting_threshold"][i] for i in train_indices],
                    "expert_policy": [expert_policy for _ in train_indices]

                },
            }
            test_data = {
                "episode": [pt_dataset["episode"][i] for i in test_indices],
                "frame_number": [pt_dataset["frame_number"][i] for i in test_indices],
                "observations": [pt_dataset["observations"][i] for i in test_indices],
                "actions": [pt_dataset["actions"][i] for i in test_indices],
                "dones": [pt_dataset["dones"][i] for i in test_indices],
                "metadata": {
                    "human_skill_label": [pt_dataset["metadata"]["human_skill_label"][i] for i in test_indices],
                    "shutter_skill_label": [pt_dataset["metadata"]["shutter_skill_label"][i] for i in test_indices],
                    "human_shooting_threshold": [pt_dataset["metadata"]["human_shooting_threshold"][i] for i in test_indices],
                    "shutter_shooting_threshold": [pt_dataset["metadata"]["shutter_shooting_threshold"][i] for i in test_indices],
                    "expert_policy": [expert_policy for _ in test_indices]
                },
            }

            eval_data = {
                "episode": [pt_dataset["episode"][i] for i in eval_indices],
                "frame_number": [pt_dataset["frame_number"][i] for i in eval_indices],
                "observations": [pt_dataset["observations"][i] for i in eval_indices],
                "actions": [pt_dataset["actions"][i] for i in eval_indices],
                "dones": [pt_dataset["dones"][i] for i in eval_indices],
                "metadata": {
                    "human_skill_label": [pt_dataset["metadata"]["human_skill_label"][i] for i in eval_indices],
                    "shutter_skill_label": [pt_dataset["metadata"]["shutter_skill_label"][i] for i in eval_indices],
                    "human_shooting_threshold": [pt_dataset["metadata"]["human_shooting_threshold"][i] for i in eval_indices],
                    "shutter_shooting_threshold": [pt_dataset["metadata"]["shutter_shooting_threshold"][i] for i in eval_indices],
                    "expert_policy": [expert_policy for _ in eval_indices]
                },
            }           
            combined_train["episode"].extend(train_data["episode"])
            combined_train["frame_number"].extend(train_data["frame_number"])
            combined_train["observations"].extend(train_data["observations"])
            combined_train["actions"].extend(train_data["actions"])
            combined_train["dones"].extend(train_data["dones"])

            combined_train["metadata"]["human_skill_label"].extend(train_data["metadata"]["human_skill_label"])
            combined_train["metadata"]["shutter_skill_label"].extend(train_data["metadata"]["shutter_skill_label"])
            combined_train["metadata"]["human_shooting_threshold"].extend(train_data["metadata"]["human_shooting_threshold"])
            combined_train["metadata"]["shutter_shooting_threshold"].extend(train_data["metadata"]["shutter_shooting_threshold"])
            combined_train["metadata"]["expert_policy"].extend(train_data["metadata"]["expert_policy"])

            combined_test["episode"].extend(test_data["episode"])
            combined_test["frame_number"].extend(test_data["frame_number"])
            combined_test["observations"].extend(test_data["observations"])
            combined_test["actions"].extend(test_data["actions"])
            combined_test["dones"].extend(test_data["dones"])

            combined_test["metadata"]["human_skill_label"].extend(test_data["metadata"]["human_skill_label"])
            combined_test["metadata"]["shutter_skill_label"].extend(test_data["metadata"]["shutter_skill_label"])
            combined_test["metadata"]["human_shooting_threshold"].extend(test_data["metadata"]["human_shooting_threshold"])
            combined_test["metadata"]["shutter_shooting_threshold"].extend(test_data["metadata"]["shutter_shooting_threshold"])
            combined_test["metadata"]["expert_policy"].extend(test_data["metadata"]["expert_policy"])

            combined_eval["episode"].extend(eval_data["episode"])
            combined_eval["frame_number"].extend(eval_data["frame_number"])
            combined_eval["observations"].extend(eval_data["observations"])
            combined_eval["actions"].extend(eval_data["actions"])
            combined_eval["dones"].extend(eval_data["dones"])

            combined_eval["metadata"]["human_skill_label"].extend(eval_data["metadata"]["human_skill_label"])
            combined_eval["metadata"]["shutter_skill_label"].extend(eval_data["metadata"]["shutter_skill_label"])
            combined_eval["metadata"]["human_shooting_threshold"].extend(eval_data["metadata"]["human_shooting_threshold"])
            combined_eval["metadata"]["shutter_shooting_threshold"].extend(eval_data["metadata"]["shutter_shooting_threshold"])
            combined_eval["metadata"]["expert_policy"].extend(eval_data["metadata"]["expert_policy"])

os.makedirs(output_dir, exist_ok=True)

torch.save(combined_train, os.path.join(output_dir, "train.pt"))
torch.save(combined_test, os.path.join(output_dir, "test.pt"))
torch.save(combined_eval, os.path.join(output_dir, "eval.pt"))

summarize_split("train", combined_train)
summarize_split("test", combined_test)
summarize_split("eval", combined_eval)

            





        
