from deep_sarsa import DeepSarsa

from consts import (
    ActionSpaces,
    DynamicsConsts,
    GameTypes,
    NaoSupportPolicies,
    RenderModes,
    RewardTypes,
)
from utils import create_env, register_env


MODEL_PATH = "experiments/baseline_run_2/models/env=competitive__support=biasedSupportLeft__reward=full.pt"
NUM_EPISODES = 3


def main():
    env_id = register_env(game_type=GameTypes.COMPETITIVE)
    env = create_env(
        env_id=env_id,
        multiprocessing=False,
        action_space=ActionSpaces.HUMAN_ONLY,
        reward_model=RewardTypes.FULL,
        render_mode=RenderModes.DISPLAY_WINDOW.value,
        support_policy=NaoSupportPolicies.BIASED_SUPPORT_LEFT,
        render=True,
        fixed_framerate=DynamicsConsts.FRAMES_PER_SECOND,
    )

    model = DeepSarsa.load(MODEL_PATH, env=env)

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
