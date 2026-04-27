import numpy as np
import argparse
import os
import random

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# Define paths relative to the script's location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")


def test_bellman_episodes(model, env_id="MsPacmanNoFrameskip-v4", num_episodes=3, seed=42):
    """
    Runs multiple game episodes to test Bellman consistency.
    For each episode, it calculates the average absolute Bellman error and total reward.
    """
    # 1. Create the environment for the test session
    print("Setting up environment for test session...")
    vec_env = make_atari_env(env_id, n_envs=1, seed=seed)
    # Transpose and stack frames to match model's expected input
    transposed_env = VecTransposeImage(vec_env)
    env = VecFrameStack(transposed_env, n_stack=4)

    # 2. Get necessary components from the model
    q_net = model.q_net
    gamma = model.gamma
    device = model.device

    # 3. Run episodes and collect results
    results = []
    print(f"Running {num_episodes} episodes to test Bellman consistency...")

    for i in range(num_episodes):
        total_bellman_error = 0
        num_steps = 0
        current_episode_reward = 0
        obs = env.reset()
        done = np.array([False])

        while not done[0]:
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs, device=device)

            # Get predicted Q-value from the main network
            with torch.no_grad():
                q_values_current = q_net(obs_tensor)

            action, _ = model.predict(obs, deterministic=True)
            q_predicted = q_values_current[0, action[0]].item()

            # Take a step in the environment
            next_obs, reward, done, _ = env.step(action)
            current_episode_reward += reward[0]

            # Calculate the Bellman target value
            target_q_value = 0.0
            if not done[0]:
                # Use the target network for the next state's Q-value
                next_obs_tensor = torch.as_tensor(next_obs, device=device)
                with torch.no_grad():
                    q_values_next = model.q_net_target(next_obs_tensor)
                    max_q_next = q_values_next.max().item()
                target_q_value = reward[0] + gamma * max_q_next
            else:
                # For the terminal state, the target is just the reward
                target_q_value = reward[0]

            # Calculate and accumulate the absolute Bellman error
            bellman_error = abs(q_predicted - target_q_value)
            total_bellman_error += bellman_error
            num_steps += 1
            obs = next_obs

        # Calculate average error for the episode
        avg_error = total_bellman_error / num_steps if num_steps > 0 else 0
        results.append((i + 1, num_steps, avg_error, current_episode_reward))
        print(
            f"Episode {i + 1} finished. Steps: {num_steps}, Reward: {current_episode_reward}, Avg Bellman Error: {avg_error:.4f}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for Bellman equation consistency.")
    parser.add_argument("--model", type=str, default="dqn", choices=["dqn"],
                        help="Model to test (only dqn is supported).")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to run for the test.")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Allowed overall average Bellman error before the test fails.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    args = parser.parse_args()

    # Load the model
    print(f"Loading DQN model from {DQN_MODEL_PATH}...")
    if not os.path.exists(DQN_MODEL_PATH):
        print(f"Error: Model file not found at {DQN_MODEL_PATH}")
        return
    # The environment is created inside the test function, so we can pass `None` here
    model = DQN.load(DQN_MODEL_PATH, env=None, device="cpu")

    # Run the test
    results = test_bellman_episodes(model, num_episodes=args.num_episodes, seed=args.seed)

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("                  Bellman Consistency Test Summary")
    print("=" * 70)
    print(f"{'Episode':<10} | {'Total Steps':<15} | {'Total Reward':<15} | {'Avg Bellman Error':<20}")
    print("-" * 70)

    total_steps_all_episodes = 0
    total_weighted_error = 0
    total_reward_all_episodes = 0

    for r in results:
        episode_num, steps, avg_error, total_reward = r
        print(f"{episode_num:<10} | {steps:<15} | {total_reward:<15.0f} | {avg_error:<20.4f}")
        total_steps_all_episodes += steps
        total_weighted_error += avg_error * steps
        total_reward_all_episodes += total_reward

    print("=" * 70)

    # Calculate the overall weighted average Bellman error
    overall_avg_error = (total_weighted_error / total_steps_all_episodes) if total_steps_all_episodes > 0 else 0
    result = "PASSED" if overall_avg_error <= args.threshold else "FAILED"

    print("\n" + "=" * 50)
    print("                 Overall Results")
    print("=" * 50)
    print(f"{'Total Episodes':<30}: {args.num_episodes:>8}")
    print(f"{'Total Steps':<30}: {total_steps_all_episodes:>8}")
    print(f"{'Total Reward':<30}: {total_reward_all_episodes:>8.0f}")
    print(f"{'Overall Avg Bellman Error':<30}: {overall_avg_error:>8.4f}")
    print(f"{'Threshold':<30}: {args.threshold:>8.4f}")
    print(f"{'Final Result':<30}: {result:>8}")
    print("=" * 50)


if __name__ == "__main__":
    main()
