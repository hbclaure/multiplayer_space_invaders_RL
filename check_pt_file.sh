python - <<'PY'
import torch
path = '/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/analysis_results/1_collect_data.py:2026-04-05_22-35-02/behavior_cloning_data.pt'
data = torch.load(path, weights_only=False)

print(type(data))
print(data.keys())
print("observations:", len(data["observations"]))
print("actions:", len(data["actions"]))
print("dones:", len(data["dones"]))
print("episodes:", len(data["episode"]))
print("first obs shape:", getattr(data["observations"][0], "shape", None))
print("first 5 actions:", data["actions"][:5])
print("unique episodes:", len(set(data["episode"])))
print("action counts:", {a: data["actions"].count(a) for a in set(data["actions"])})
print("human labels:", len(data["metadata"]["human_skill_label"]))
print("shutter labels:", len(data["metadata"]["shutter_skill_label"]))
PY
