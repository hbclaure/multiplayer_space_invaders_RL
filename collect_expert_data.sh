#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

NUM_EPISODES=10
ADJUST_PLAYER_SHOOTING="both"
BASE_EXP_NAME="collect_data"
EXTRA_ARGS=()
POLICIES=()

usage() {
  cat <<'EOF'
Usage:
  ./collect_expert_data.sh [options] policy1 [policy2 ...]

Examples:
  ./collect_expert_data.sh equalizeScoresNeutral equalizeScores
  ./collect_expert_data.sh --num-episodes 25 --exp-name bc_seed1 biasedSupportLeft biasedSupportRight
  ./collect_expert_data.sh --adjust-player-shooting both -- equalizeHistory equalizeHistoryEven

Options:
  --num-episodes N              Number of episodes per policy (default: 10)
  --adjust-player-shooting VAL  human|shutter|both (default: both)
  --exp-name NAME               Run suffix used inside each policy folder (default: collect_data)
  --render                      Pass through to 1_collect_data.py
  --human-weak-player           Pass through to 1_collect_data.py
  --shutter-weak-player         Pass through to 1_collect_data.py
  --independent-victory-score-threshold N
                                Pass through to 1_collect_data.py
  --help                        Show this message

Notes:
  - Each policy run is saved under:
    analysis_results/behavior_cloning/<support-policy>/<exp-name>/
  - Policy names are the same values accepted by --support-policy in 1_collect_data.py
EOF
}

while (($# > 0)); do
  case "$1" in
    --num-episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --adjust-player-shooting)
      ADJUST_PLAYER_SHOOTING="$2"
      shift 2
      ;;
    --exp-name)
      BASE_EXP_NAME="$2"
      shift 2
      ;;
    --render|--human-weak-player|--shutter-weak-player)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --independent-victory-score-threshold)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while (($# > 0)); do
        POLICIES+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      POLICIES+=("$1")
      shift
      ;;
  esac
done

if ((${#POLICIES[@]} == 0)); then
  echo "No support policies provided." >&2
  usage >&2
  exit 1
fi

for policy in "${POLICIES[@]}"; do
  echo "Running data collection for policy: $policy"
  python 1_collect_data.py \
    --nao-controller heuristic \
    --support-policy "$policy" \
    --adjust-player-shooting "$ADJUST_PLAYER_SHOOTING" \
    --num-episodes "$NUM_EPISODES" \
    --exp-name "$BASE_EXP_NAME" \
    "${EXTRA_ARGS[@]}"
done
