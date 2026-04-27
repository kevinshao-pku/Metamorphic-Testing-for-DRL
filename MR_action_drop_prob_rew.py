import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
import os
import random

# Define paths relative to the script's location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")

# The 'NOOP' action for Atari environments is 0
NOOP_ACTION = 0


def play_game_session(model, num_episodes=3, seed=42, drop_probability=0.0):
    """
    Runs a game session, potentially dropping actions (replacing with NOOP).
    Returns a list of rewards, one for each episode.
    """
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []

    for i in range(num_episodes):
        obs = env.reset()
        terminated = False
        current_episode_reward = 0
        while not terminated:
            if random.random() < drop_probability:
                action = np.array([NOOP_ACTION])
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, rewards, terminated, info = env.step(action)
            current_episode_reward += rewards[0]
        print(f"Episode {i + 1} finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for action drop robustness.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--drop_probability", type=float, default=0.1, help="Probability of dropping an action (forcing NOOP).")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Allowed percentage change before the test fails.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.model == "dqn":
        print("Loading DQN model...")
        model = DQN.load(DQN_MODEL_PATH, device="cpu")
    else:
        print("Loading PPO model...")
        model = PPO.load(PPO_MODEL_PATH, device="cpu")

    print(f"\nRunning original case (no actions dropped)...")
    original_rewards = play_game_session(model, seed=args.seed, drop_probability=0.0)
    original_reward = sum(original_rewards)

    print(f"\nRunning mutated case (dropping actions with {args.drop_probability:.0%} probability)...")
    mutated_rewards = play_game_session(model, seed=args.seed, drop_probability=args.drop_probability)
    mutated_reward = sum(mutated_rewards)

    if original_reward == 0:
        reward_change_percent = 0 if mutated_reward == 0 else float('inf')
    else:
        # Use raw difference for display, absolute for logic
        reward_change_percent = 100 * (mutated_reward - original_reward) / abs(original_reward)

    reward_change_abs = abs(reward_change_percent)
    result = "PASSED" if reward_change_abs <= args.threshold else "FAILED"

    # Format all values for alignment
    prob_str = f"{args.drop_probability:.0%}"
    reward_change_str = f"{reward_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n--- ACTION DROP TEST RESULTS ---")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Drop Probability:      {prob_str:>8}")
    print(f"Original Reward Total: {original_reward:>8.2f}")
    print(f"Mutated Reward Total:  {mutated_reward:>8.2f}")
    print(f"Reward Change:         {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")


if __name__ == "__main__":
    main()