import numpy as np
import argparse
import os
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def play_game_session(model, num_episodes=3, seed=42, drop_interval=None):
    """
    Runs a game session, optionally applying a temporal stutter every k steps.
    The stutter is created by duplicating the second-to-last frame into the last frame.
    Returns a list of rewards, one for each episode.
    """
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []

    for i in range(num_episodes):
        obs = env.reset()
        terminated = np.array([False])
        current_episode_reward = 0
        step_counter = 0
        while not terminated[0]:
            final_obs = np.array(obs)

            # If drop_interval is provided, apply stutter every k steps
            if drop_interval and step_counter > 0 and step_counter % drop_interval == 0:
                # This simulates the agent "losing an input" by not seeing the newest frame.
                final_obs[..., 3] = final_obs[..., 2]

            action, _ = model.predict(final_obs, deterministic=True)
            obs, rewards, terminated, info = env.step(action)
            current_episode_reward += rewards[0]
            step_counter += 1

        case = f"Mutated (k={drop_interval})" if drop_interval else "Original"
        print(f"Episode {i + 1} ({case}) finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for performance change with periodic temporal "
                                                 "stutter.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes for each run.")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Allowed percentage performance drop before the test fails.")
    parser.add_argument("--drop_interval", type=int, required=True,
                        help="The agent loses an input every k steps.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.model == "dqn":
        model_path = DQN_MODEL_PATH
        model_class = DQN
    else:
        model_path = PPO_MODEL_PATH
        model_class = PPO

    print(f"Loading {args.model.upper()} model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model = model_class.load(model_path, device="cpu")

    print(f"\nRunning original case (no stutter)...")
    original_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes)
    original_reward_total = sum(original_rewards)

    print(f"\nRunning mutated case (stutter every {args.drop_interval} steps)...")
    mutated_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes,
                                        drop_interval=args.drop_interval)
    mutated_reward_total = sum(mutated_rewards)

    if original_reward_total == 0:
        performance_change_percent = 0 if mutated_reward_total == 0 else float('-inf')
    else:
        performance_change_percent = 100 * (mutated_reward_total - original_reward_total) / abs(original_reward_total)

    performance_drop_abs = abs(performance_change_percent)
    result = "PASSED" if performance_drop_abs <= args.threshold else "FAILED"

    reward_change_str = f"{performance_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n" + "-" * 55)
    print("        FRAME DROP REWARD-BASED RESULTS")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Drop Interval (k):     {args.drop_interval:>8}")
    print(f"Original Reward Total: {original_reward_total:>8.0f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.0f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("-" * 55)


if __name__ == "__main__":
    main()