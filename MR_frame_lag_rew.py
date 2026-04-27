import numpy as np
import argparse
import os
import random
import collections
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def play_game_session(model, num_episodes=3, seed=42, lag_amount=0):
    """
    Runs a game session, applying a frame lag of k steps.
    The agent always receives an observation from 'lag_amount' steps ago.
    Returns a list of rewards, one for each episode.
    """
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []

    for i in range(num_episodes):
        obs = env.reset()
        terminated = np.array([False])
        current_episode_reward = 0

        # Buffer to store past observations
        obs_buffer = collections.deque(maxlen=lag_amount + 1)

        while not terminated[0]:
            current_obs = np.array(obs)
            obs_buffer.append(current_obs)

            # If lag is enabled and the buffer is full, use the oldest observation
            if 0 < lag_amount < len(obs_buffer):
                final_obs = obs_buffer[0]
            else:
                # Otherwise (no lag or buffer not full), use the current observation
                final_obs = current_obs

            action, _ = model.predict(final_obs, deterministic=True)
            obs, rewards, terminated, info = env.step(action)
            current_episode_reward += rewards[0]

        case = f"Mutated (lag={lag_amount})" if lag_amount > 0 else "Original"
        print(f"Episode {i + 1} ({case}) finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for performance change with frame lag.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes for each run.")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Allowed percentage performance drop before the test fails.")
    parser.add_argument("--lag_amount", type=int, default=2,
                        help="The number of frames the agent's observation lags behind.")
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

    print(f"\nRunning original case (no lag)...")
    original_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes, lag_amount=0)
    original_reward_total = sum(original_rewards)

    print(f"\nRunning mutated case (lag by {args.lag_amount} steps)...")
    mutated_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes,
                                        lag_amount=args.lag_amount)
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
    print("        FRAME LAG REWARD-BASED RESULTS")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Lag Amount (k):        {args.lag_amount:>8}")
    print(f"Original Reward Total: {original_reward_total:>8.0f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.0f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("-" * 55)


if __name__ == "__main__":
    main()
