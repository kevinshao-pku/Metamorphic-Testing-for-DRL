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


def test_brightness_invariance(model, seed=42, num_episodes=3, brightness_offset=30, threshold=5.0):
    """
    Tests if the model's actions are invariant to a brightness offset.
    Compares the action taken on an original observation vs. a brightened one.
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
            total_steps += 1

            # Get the action for the original observation
            original_action, _ = model.predict(obs, deterministic=True)

            # Create the mutated observation (brighter)
            obs_np = np.array(obs)
            modified_obs_int = obs_np.astype(np.int16) + brightness_offset
            final_obs = np.clip(modified_obs_int, 0, 255).astype(np.uint8)

            # Get the action for the mutated observation
            mutated_action, _ = model.predict(final_obs, deterministic=True)

            # Check for divergence
            if original_action[0] != mutated_action[0]:
                divergent_steps += 1

            # Step the environment with the original action to continue the trajectory
            obs, _, terminated, _ = env.step(original_action)

        print(f"Episode {i + 1} finished.")

    env.close()

    # --- METRICS CALCULATION AND REPORTING ---
    if total_steps == 0:
        divergence_rate = 0
    else:
        divergence_rate = (divergent_steps / total_steps) * 100

    result = "PASSED" if divergence_rate <= threshold else "FAILED"

    # Format all values for alignment
    brightness_str = f"{brightness_offset:+d}"
    divergence_rate_str = f"{divergence_rate:+0.2f}%"
    threshold_str = f"{threshold:.2f}%"

    # --- USE CLEAR, ACTION-BASED LABELS ---
    print("\n" + "=" * 40)
    print("    BRIGHTNESS TEST RESULTS (ACTION)")
    print(f"Seed: {seed}")
    print(f"Model:                 {model.__class__.__name__:>8}")
    print(f"Brightness Offset:     {brightness_str:>8}")
    print(f"Total Steps:           {total_steps:>8}")
    print(f"Divergent Steps:       {divergent_steps:>8}")
    print(f"Divergence Rate:       {divergence_rate_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Metamorphic test for brightness invariance (action-based).")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--brightness_offset", type=int, default=30, help="Integer brightness offset to apply.")
    parser.add_argument("--threshold", type=float, default=5.0, help="Allowed divergence rate before the test fails.")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for the environment.')
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to run.")
    args = parser.parse_args()

    if args.model == "dqn":
        print("Loading DQN model...")
        model = DQN.load(DQN_MODEL_PATH, device="cpu")
    else:
        print("Loading PPO model...")
        model = PPO.load(PPO_MODEL_PATH, device="cpu")

    test_brightness_invariance(
        model,
        seed=args.seed,
        num_episodes=args.num_episodes,
        brightness_offset=args.brightness_offset,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()