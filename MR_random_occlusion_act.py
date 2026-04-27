import numpy as np
import argparse
import os
import random
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")

def apply_random_occlusion(obs, occlusion_size):
    """
    Applies a random occlusion to a single observation.
    """
    occluded_obs = np.copy(obs)
    frame_height, frame_width = occluded_obs.shape[1], occluded_obs.shape[2]

    if occlusion_size > 0:
        # Choose a random top-left corner for the occlusion
        occlusion_x = random.randint(0, frame_width - occlusion_size)
        occlusion_y = random.randint(0, frame_height - occlusion_size)

        # Apply the occlusion to all frames in the stack
        occluded_obs[:, occlusion_y:occlusion_y + occlusion_size, occlusion_x:occlusion_x + occlusion_size, :] = 0

    return occluded_obs

def test_occlusion_invariance(model, env_id="MsPacmanNoFrameskip-v4", seed=42, threshold=15.0, occlusion_size=20):
    """
    Runs a single game episode to test if the agent's action is invariant
    to a random occlusion.
    """
    # Set all relevant random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_atari_env(env_id, n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    obs = env.reset()
    terminated = [False]
    total_steps = 0
    divergent_steps = 0

    while not terminated[0]:
        action_original, _ = model.predict(obs, deterministic=True)

        mutated_obs = apply_random_occlusion(obs, occlusion_size)
        action_mutated, _ = model.predict(mutated_obs, deterministic=True)

        if action_original[0] != action_mutated[0]:
            divergent_steps += 1

        obs, _, terminated, _ = env.step(action_original)
        total_steps += 1

    env.close()

    divergence_pct = (divergent_steps / total_steps) * 100 if total_steps > 0 else 0
    result = "PASSED" if divergence_pct <= threshold else "FAILED"
    divergence_rate_str = f"{divergence_pct:.2f}%"
    threshold_str = f"{threshold:.2f}%"

    # --- UNIFIED CONSOLE OUTPUT ---
    print("\n" + "=" * 40)
    print(" RANDOM OCCLUSION TEST RESULTS (ACTION)")
    print(f"Seed: {seed}")
    print(f"Model:                 {model.__class__.__name__:>8}")
    print(f"Occlusion Size:        {f'{occlusion_size}x{occlusion_size}':>8}")
    print(f"Total Steps:           {total_steps:>8}")
    print(f"Divergent Steps:       {divergent_steps:>8}")
    print(f"Divergence Rate:       {divergence_rate_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Test model's invariance to random occlusion in Ms. Pac-Man.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
    parser.add_argument("--threshold", type=float, default=15.0, help="Threshold for passing the test.")
    parser.add_argument("--occlusion_size", type=int, default=20, help="The size of the random occlusion square.")
    args = parser.parse_args()

    if args.model == "dqn":
        model_path = DQN_MODEL_PATH
        model_class = DQN
    else: # ppo
        model_path = PPO_MODEL_PATH
        model_class = PPO

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model = model_class.load(model_path, device="cpu")

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    test_occlusion_invariance(model, seed=seed, threshold=args.threshold, occlusion_size=args.occlusion_size)


if __name__ == '__main__':
    main()