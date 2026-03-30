from stable_baselines3 import DQN

from deep_sarsa import DeepSarsa
import argparse
import pprint
from consts import (
    ActionSpaces,
    DynamicsConsts,
    GameTypes,
    NaoSupportPolicies,
    RenderModes,
    RewardTypes,
)
from utils import create_env, register_env

def parse_args():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning Model testing")
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
    return parser.parse_args()

MODEL_PATH = "experiments/testing_feature/models/env=competitive__support=equalSupport__reward=naoFairness.pt"
NUM_EPISODES = 3


def load_trained_model(model_path, env):
    try:
        return DQN.load(model_path, env=env)
    except Exception:
        return DeepSarsa.load(model_path, env=env)


def main():
    args = parse_args()
    print("Running train.py with the following arguments:")
    pprint.pprint(args)
    env_id = register_env(game_type=GameTypes.COMPETITIVE)
    env = create_env(
        env_id=env_id,
        multiprocessing=False,
        action_space=ActionSpaces.NAO_ONLY,
        reward_model=RewardTypes.NAO_FAIRNESS,
        render_mode=RenderModes.DISPLAY_WINDOW.value,
        support_policy=NaoSupportPolicies.EQUAL_SUPPORT,
        render=True,
        fixed_framerate=DynamicsConsts.FRAMES_PER_SECOND,
        shutter_weak_player_flag = args.shutter_weak_player ,
        human_weak_player_flag = args.human_weak_player,

    )

    model = load_trained_model(MODEL_PATH, env=env)

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
