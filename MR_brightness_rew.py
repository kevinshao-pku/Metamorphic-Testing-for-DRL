import numpy as np
import argparse
import os
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Define paths relative to the script's location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")

# The 'NOOP' action for Atari environments is 0
NOOP_ACTION = 0


def play_game_session(model, num_episodes=3, seed=42, brightness_offset=0):
    """
    Runs a game session, applying a brightness offset to the observations.
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
            # Apply the brightness offset to the observation
            if brightness_offset != 0:
                obs_np = np.array(obs)
                modified_obs_int = obs_np.astype(np.int16) + brightness_offset
                final_obs = np.clip(modified_obs_int, 0, 255).astype(np.uint8)
            else:
                final_obs = obs

            action, _ = model.predict(final_obs, deterministic=True)
            obs, rewards, terminated, info = env.step(action)
            current_episode_reward += rewards[0]
        print(f"Episode {i+1} finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for brightness invariance.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--brightness_offset", type=int, default=30, help="Integer brightness offset to apply.")
    parser.add_argument("--threshold", type=float, default=15.0, help="Allowed percentage change before the test fails.")
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

    print(f"\nRunning original case (brightness offset: 0)...")
    original_rewards = play_game_session(model, seed=args.seed, brightness_offset=0)
    original_reward_total = sum(original_rewards)

    print(f"\nRunning mutated case (brightness offset: {args.brightness_offset:+d})...")
    mutated_rewards = play_game_session(model, seed=args.seed, brightness_offset=args.brightness_offset)
    mutated_reward_total = sum(mutated_rewards)

    if original_reward_total == 0:
        reward_change_percent = 0 if mutated_reward_total == 0 else float('inf')
    else:
        # Use raw difference for display, absolute for logic
        reward_change_percent = 100 * (mutated_reward_total - original_reward_total) / abs(original_reward_total)

    reward_change_abs = abs(reward_change_percent)
    result = "PASSED" if reward_change_abs <= args.threshold else "FAILED"

    # Format all values for alignment
    brightness_str = f"{args.brightness_offset:+d}"
    reward_change_str = f"{reward_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n" + "=" * 35)
    print("    BRIGHTNESS TEST RESULTS")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Brightness Offset:     {brightness_str:>8}")
    print(f"Original Reward Total: {original_reward_total:>8.2f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.2f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 35)


if __name__ == "__main__":
    main()