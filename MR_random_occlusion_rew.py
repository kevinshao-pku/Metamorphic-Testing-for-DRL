import numpy as np
import argparse
import os
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import matplotlib.pyplot as plt

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def play_game_session(model, num_episodes=3, seed=42, apply_occlusion=False, occlusion_size=10, save_visualization=False):
    """
    Runs a game session, optionally applying a random occlusion to observations.
    Returns a list of rewards, one for each episode.
    """
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []
    visualization_saved = False

    for i in range(num_episodes):
        obs = env.reset()
        terminated = np.array([False])
        current_episode_reward = 0
        step_counter = 0
        while not terminated[0]:
            final_obs = np.array(obs)

            if apply_occlusion:
                # Determine the dimensions of the observation space
                frame_height, frame_width = final_obs.shape[1], final_obs.shape[2]

                # Choose a random top-left corner for the occlusion
                # Ensure the occlusion is fully within the frame
                occlusion_x = random.randint(0, frame_width - occlusion_size)
                occlusion_y = random.randint(0, frame_height - occlusion_size)

                # Apply the occlusion to all frames in the stack
                final_obs[:, occlusion_y:occlusion_y + occlusion_size, occlusion_x:occlusion_x + occlusion_size, :] = 0

                # --- Visualization Logic ---
                if save_visualization and not visualization_saved and step_counter >= 500:
                    original_frame = np.array(obs)[0, :, :, 3]
                    plt.imsave("original_frame.png", original_frame, cmap='gray')

                    mutated_frame_viz = final_obs[0, :, :, 3]
                    plt.imsave("mutated_frame.png", mutated_frame_viz, cmap='gray')

                    print(
                        f"Saved visualization frames (occlusion size {occlusion_size}x{occlusion_size}): "
                        f"'original_frame.png' and 'mutated_frame.png'")
                    visualization_saved = True
                # --- End Visualization Logic ---

            action, _ = model.predict(final_obs, deterministic=True)
            obs, rewards, terminated, info = env.step(action)
            current_episode_reward += rewards[0]
            step_counter += 1

        case = "Mutated" if apply_occlusion else "Original"
        print(f"Episode {i + 1} ({case}) finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for performance change with random occlusion.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes for each run.")
    parser.add_argument("--threshold", type=float, default=25.0, # Increased threshold slightly for this test
                        help="Allowed percentage performance drop before the test fails.")
    parser.add_argument("--save_frame_visualization", action="store_true",
                        help="Save a before/after image of the occlusion mutation.")
    parser.add_argument("--occlusion_size", type=int, default=20,
                        help="The size (width and height) of the random occlusion square.")
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

    print(f"\nRunning original case (no occlusion)...")
    original_rewards = play_game_session(model, seed=args.seed, num_episodes=args.num_episodes, apply_occlusion=False)
    original_reward_total = sum(original_rewards)

    print(f"\nRunning mutated case (with random occlusion)...")
    mutated_rewards = play_game_session(model,  seed=args.seed, num_episodes=args.num_episodes, apply_occlusion=True,
                                        occlusion_size=args.occlusion_size,
                                        save_visualization=args.save_frame_visualization)
    mutated_reward_total = sum(mutated_rewards)

    if original_reward_total == 0:
        performance_change_percent = 0 if mutated_reward_total == 0 else float('-inf')
    else:
        performance_change_percent = 100 * (mutated_reward_total - original_reward_total) / abs(original_reward_total)

    performance_drop_abs = abs(performance_change_percent)
    result = "PASSED" if performance_drop_abs <= args.threshold else "FAILED"

    reward_change_str = f"{performance_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n" + "-" * 50)
    print("      RANDOM OCCLUSION PERFORMANCE RESULTS")
    print("-" * 50)
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Occlusion Size:        {f'{args.occlusion_size}x{args.occlusion_size}':>8}")
    print(f"Original Reward Total: {original_reward_total:>8.0f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.0f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("-" * 50)


if __name__ == "__main__":
    main()