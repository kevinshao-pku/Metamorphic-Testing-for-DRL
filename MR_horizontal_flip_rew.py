import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
import os
import matplotlib.pyplot as plt

# This dictionary maps actions to their horizontal counterparts.
ACTION_MAP = {
    0: 0,  # NOOP -> NOOP
    1: 1,  # UP -> UP
    2: 3,  # RIGHT -> LEFT
    3: 2,  # LEFT -> RIGHT
    4: 4,  # DOWN -> DOWN
    5: 6,  # UPRIGHT -> UPLEFT
    6: 5,  # UPLEFT -> UPRIGHT
    7: 8,  # DOWNRIGHT -> DOWNLEFT
    8: 7  # DOWNLEFT -> DOWNRIGHT
}

# Define paths relative to the script's location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def play_game_session(model, num_episodes=3, seed=42, flip_screen=False, save_visualization=False, score_rows=15):
    """
    Runs a game session, optionally flipping the observations horizontally.
    Returns a list of rewards, one for each episode.
    """
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []
    visualization_saved = False

    for i in range(num_episodes):
        obs = env.reset()
        terminated = False
        current_episode_reward = 0
        step_counter = 0
        while not terminated:
            final_obs = obs
            if flip_screen:
                obs_np = np.array(obs)

                # --- Visualization Logic ---
                if save_visualization and not visualization_saved and step_counter >= 500:
                    original_frame = obs_np[0, :, :, -1]  # Get the most recent frame
                    plt.imsave("original_frame.png", original_frame, cmap='gray')

                    game_part_viz = original_frame[:-score_rows, :]
                    score_part_viz = original_frame[-score_rows:, :]
                    flipped_game_part_viz = np.flip(game_part_viz, axis=1)
                    mutated_frame_viz = np.concatenate((flipped_game_part_viz, score_part_viz), axis=0)
                    plt.imsave("mutated_frame.png", mutated_frame_viz, cmap='gray')

                    print(f"\nSaved visualization frames: 'original_frame.png' and 'mutated_frame.png'")
                    visualization_saved = True
                # --- End Visualization Logic ---

                # Apply the partial flip for the agent's action
                game_screen = obs_np[:, :-score_rows, :, :]
                score_bar = obs_np[:, -score_rows:, :, :]
                flipped_game_screen = np.flip(game_screen, axis=2)  # Flip along the width axis
                final_obs = np.concatenate((flipped_game_screen, score_bar), axis=1).copy()
            else:
                final_obs = obs

            action, _ = model.predict(final_obs, deterministic=True)

            # If the screen was flipped, the action must be mapped back to the original coordinate system
            if flip_screen:
                mapped_action = ACTION_MAP.get(action[0])
                final_action = np.array([mapped_action])
            else:
                final_action = action

            obs, rewards, terminated, info = env.step(final_action)
            current_episode_reward += rewards[0]
            step_counter += 1

        print(f"Episode {i + 1} finished. Reward: {current_episode_reward}")
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for horizontal flip invariance.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Allowed percentage change before the test fails.")
    parser.add_argument("--save_frame_visualization", action="store_true",
                        help="Save a before/after image of the horizontal flip mutation.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    args = parser.parse_args()

    if args.model == "dqn":
        print("Loading DQN model...")
        model = DQN.load(DQN_MODEL_PATH, device="cpu")
    else:
        print("Loading PPO model...")
        model = PPO.load(PPO_MODEL_PATH, device="cpu")

    print(f"\nRunning original case (screen not flipped)...")
    original_rewards = play_game_session(model, seed=args.seed, flip_screen=False)
    original_reward_total = sum(original_rewards)

    print(f"\nRunning mutated case (screen flipped)...")
    mutated_rewards = play_game_session(model, seed=args.seed, flip_screen=True, save_visualization=args.save_frame_visualization)
    mutated_reward_total = sum(mutated_rewards)

    if original_reward_total == 0:
        reward_change_percent = 0 if mutated_reward_total == 0 else float('inf')
    else:
        reward_change_percent = 100 * (mutated_reward_total - original_reward_total) / abs(original_reward_total)

    reward_change_abs = abs(reward_change_percent)
    result = "PASSED" if reward_change_abs <= args.threshold else "FAILED"

    reward_change_str = f"{reward_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n" + "=" * 35)
    print(" HORIZONTAL FLIP TEST RESULTS")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Original Reward Total: {original_reward_total:>8.2f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.2f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 35)


if __name__ == "__main__":
    main()