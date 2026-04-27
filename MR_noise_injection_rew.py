import numpy as np
import argparse
import os
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import matplotlib.pyplot as plt

# Define paths relative to the script's location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def play_game_session(model, num_episodes=3, seed=42, inject_noise=False, noise_level=10.0, save_visualization=False):
    """
    Runs a game session, optionally injecting Gaussian noise into observations.
    Returns a list of rewards, one for each episode.
    """
    # Create the environment. `make_atari_env` handles resizing to 84x84 and grayscale.
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    # `VecFrameStack` stacks 4 frames. The final observation shape is (1, 84, 84, 4).
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []
    visualization_saved = False

    for i in range(num_episodes):
        obs = env.reset()
        terminated = np.array([False])
        current_episode_reward = 0
        step_counter = 0
        while not terminated[0]:
            final_obs = obs  # Default to the original observation

            # Apply the noise injection mutation if enabled
            if inject_noise:
                obs_array = np.array(obs, dtype=np.float32)

                # Generate Gaussian noise
                noise = np.random.normal(0, noise_level, obs_array.shape).astype(np.float32)

                # Add noise and clip the values to the valid range [0, 255]
                noisy_obs = obs_array + noise
                noisy_obs = np.clip(noisy_obs, 0, 255)

                final_obs = noisy_obs.astype(np.uint8)

                # --- Visualization Logic ---
                if save_visualization and not visualization_saved and step_counter >= 500:
                    # Get the last frame (most recent) for visualization.
                    original_frame = np.array(obs)[0, :, :, 3]
                    plt.imsave("original_frame.png", original_frame, cmap='gray')

                    # Create a mutated copy for visualization
                    mutated_frame_viz = final_obs[0, :, :, 3]
                    plt.imsave("mutated_frame.png", mutated_frame_viz, cmap='gray')

                    print(f"Saved visualization frames (noise level {noise_level}): "
                          f"'original_frame.png' and 'mutated_frame.png'")
                    visualization_saved = True
                # --- End Visualization Logic ---

            action, _ = model.predict(final_obs, deterministic=True)
            obs, rewards, terminated, info = env.step(action)
            current_episode_reward += rewards[0]
            step_counter += 1

        case = "Mutated" if inject_noise else "Original"
        print(f"Episode {i + 1} ({case}) finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for performance change with noise injection.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes for each run.")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Allowed percentage performance drop before the test fails.")
    parser.add_argument("--save_frame_visualization", action="store_true",
                        help="Save a before/after image of the noise injection mutation.")
    parser.add_argument("--noise_level", type=float, default=15.0,
                        help="Standard deviation of the Gaussian noise to inject.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load the selected model
    if args.model == "dqn":
        model_path = DQN_MODEL_PATH
        model_class = DQN
    else:  # ppo
        model_path = PPO_MODEL_PATH
        model_class = PPO

    print(f"Loading {args.model.upper()} model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model = model_class.load(model_path, device="cpu")

    # Run original case
    print(f"\nRunning original case (no noise)...")
    original_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes, inject_noise=False)
    original_reward_total = sum(original_rewards)

    # Run mutated case
    print(f"\nRunning mutated case (with noise injection)...")
    mutated_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes, inject_noise=True,
                                        noise_level=args.noise_level, save_visualization=args.save_frame_visualization)
    mutated_reward_total = sum(mutated_rewards)

    # Calculate and evaluate performance change
    if original_reward_total == 0:
        performance_change_percent = 0 if mutated_reward_total == 0 else float('-inf')
    else:
        performance_change_percent = 100 * (mutated_reward_total - original_reward_total) / abs(original_reward_total)

    performance_drop_abs = abs(performance_change_percent)
    result = "PASSED" if performance_drop_abs <= args.threshold else "FAILED"

    # --- Format all values for alignment ---
    reward_change_str = f"{performance_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n" + "=" * 50)
    print("      NOISE INJECTION PERFORMANCE RESULTS")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Noise Level (Std Dev): {args.noise_level:>8.1f}")
    print(f"Original Reward Total: {original_reward_total:>8.2f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.2f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 50)


if __name__ == "__main__":
    main()