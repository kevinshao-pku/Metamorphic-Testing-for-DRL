import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
import os
import random

# This dictionary maps actions to their horizontal counterparts.
# When the game screen is flipped horizontally, the model's intended action
# in the flipped space needs to be mapped back to the original frame of reference.
# For example, an action 'RIGHT' in the flipped view is actually 'LEFT' in the original view.
ACTION_MAP = {
    # action_from_flipped_obs: action_in_original_obs
    0: 0,  # NOOP -> NOOP
    1: 1,  # UP -> UP
    2: 3,  # RIGHT -> LEFT
    3: 2,  # LEFT -> RIGHT
    4: 4,  # DOWN -> DOWN
    5: 6,  # UPRIGHT -> UPLEFT
    6: 5,  # UPLEFT -> UPRIGHT
    7: 8,  # DOWNRIGHT -> DOWNLEFT
    8: 7   # DOWNLEFT -> DOWNRIGHT
}

# Define paths relative to the script's location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def test_symmetry_invariance(model, seed=42, num_episodes=3, threshold=5.0):
    """
    Tests if the model's actions are invariant to horizontal flipping.
    Compares the action taken on an original observation vs. a flipped one.
    """
    # --- Comprehensive Seeding for Reproducibility ---
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    total_steps = 0
    divergent_steps = 0

    for i in range(num_episodes):
        obs = env.reset()
        terminated = False
        print(f"Episode {i + 1} started.")
        while not terminated:
            # 1. Get action for the original observation
            action_original, _ = model.predict(obs, deterministic=True)

            # 2. For manipulation, convert LazyFrames to a NumPy array
            obs_np = np.array(obs)

            # 3. Flip each of the 4 frames in the stack horizontally (axis=3 is the width dimension)
            flipped_obs_np = np.flip(obs_np, axis=3).copy()

            # 4. Get action for the flipped observation
            action_flipped, _ = model.predict(flipped_obs_np, deterministic=True)

            # 5. Map the flipped action back to the original frame of reference
            action_original_item = action_original[0]
            action_flipped_item = action_flipped[0]
            mapped_action = ACTION_MAP.get(action_flipped_item)

            # 6. Check for divergence
            if action_original_item != mapped_action:
                divergent_steps += 1

            # 7. Step the environment with the original action
            obs, _, terminated, _ = env.step(action_original)
            total_steps += 1

        print(f"Episode {i + 1} finished.")

    env.close()

    # --- METRICS CALCULATION AND REPORTING ---
    if total_steps == 0:
        divergence_rate = 0
    else:
        divergence_rate = (divergent_steps / total_steps) * 100

    result = "PASSED" if divergence_rate <= threshold else "FAILED"

    # Format all values for alignment
    divergence_rate_str = f"{divergence_rate:.2f}%"
    threshold_str = f"{threshold:.2f}%"

    # --- USE CLEAR, ACTION-BASED LABELS ---
    print("\n" + "=" * 40)
    print("    SYMMETRY TEST RESULTS (ACTION)")
    print(f"Seed: {seed}")
    print(f"Model:                 {model.__class__.__name__:>8}")
    print(f"Total Steps:           {total_steps:>8}")
    print(f"Divergent Steps:       {divergent_steps:>8}")
    print(f"Divergence Rate:       {divergence_rate_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 40)


def main():
    """
    Main function to run the symmetry invariance test.
    """
    parser = argparse.ArgumentParser(description="Test model's symmetry invariance in Ms. Pac-Man (action-based).")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Allowed divergence rate before the test fails.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to run for evaluation.')
    args = parser.parse_args()

    # Load the selected model
    if args.model == "dqn":
        model_path = DQN_MODEL_PATH
        print("Loading DQN model...")
        model = DQN.load(model_path, device="cpu")
    else:
        model_path = PPO_MODEL_PATH
        print("Loading PPO model...")
        model = PPO.load(model_path, device="cpu")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    test_symmetry_invariance(
        model,
        seed=args.seed,
        num_episodes=args.num_episodes,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()