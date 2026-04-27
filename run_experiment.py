import argparse
import subprocess
import re
import pandas as pd
import numpy as np
from datetime import datetime
import random
import os


def parse_mr_output(output):
    """
    Parses the text output from the MR script to extract key metrics.
    This parser is now flexible to handle both reward-based (rew) and action-based (act) tests.
    """
    results = {}
    try:
        results['seed'] = int(re.search(r"Seed:\s*(\d+)", output).group(1))
        results['result'] = re.search(r"Result:\s*(\w+)", output).group(1)

        # Check for action-based test labels first
        if "Divergence Rate" in output:
            results['metric_type'] = 'action'
            results['total_steps'] = int(re.search(r"Total Steps:\s*(\d+)", output).group(1))
            results['divergent_steps'] = int(re.search(r"Divergent Steps:\s*(\d+)", output).group(1))
            change_pct_match = re.search(r"Divergence Rate:\s*([+\-\d\.]+)%", output)
            results['change_pct'] = float(change_pct_match.group(1))
            results['metric_label'] = 'Divergence'  # Label for clear logging

        # Fallback to reward-based test labels
        elif "Original Reward Total" in output:
            results['metric_type'] = 'reward'
            results['baseline_reward'] = float(re.search(r"Original Reward Total:\s*([-\d\.]+)", output).group(1))
            results['mutated_reward'] = float(re.search(r"Mutated Reward Total:\s*([-\d\.]+)", output).group(1))
            # This regex handles both "Reward Change" and "Performance Change" for full compatibility
            change_pct_match = re.search(r"(?:Reward|Performance) Change:\s*([+\-\d\.]+)%", output)
            results['change_pct'] = float(change_pct_match.group(1))
            results['metric_label'] = 'Reward Change'  # Label for clear logging

        else:
            raise ValueError("Output format not recognized as action-based or reward-based.")

        return results
    except (AttributeError, ValueError) as e:
        print(f"\n--- ERROR: Could not parse script output. ---")
        print(f"Error details: {e}")
        print("Please ensure the test script prints results in a recognized format.")
        print("--- Captured Output ---")
        print(output)
        print("-----------------------\n")
        return None


def main():
    """
    Main orchestrator to run a metamorphic test script multiple times with random seeds.
    """
    parser = argparse.ArgumentParser(
        description="Run a metamorphic test script multiple times and aggregate the results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("test_script", type=str, help="The MR test script to execute (e.g., MR_brightness_rew.py).")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of times to repeat the experiment.")
    parser.add_argument("--master_seed", type=int, default=11037,
                        help="Seed for the random number generator to produce a consistent set of seeds for all runs."
                             "If not provided, seeds will be different each time.")

    args, passthrough_args = parser.parse_known_args()

    if args.master_seed is not None:
        print(f"Using Master Seed: {args.master_seed} to generate a reproducible sequence of run seeds.")
        random.seed(args.master_seed)
    else:
        print("WARNING: No Master Seed provided. The sequence of seeds for this experiment will not be reproducible.")

    print(f"\nOrchestrator: Starting experiment...")
    print(f"  -> Test Script: {args.test_script}")
    print(f"  -> Number of Runs: {args.num_runs}")
    print(f"  -> Passthrough Args: {passthrough_args}")
    print("-" * 50)

    all_runs_data = []

    for i in range(args.num_runs):
        run_seed = random.randint(0, 2 ** 32 - 1)
        print(f"\n[RUN {i + 1}/{args.num_runs}] Using Seed: {run_seed}")

        command = ["python", args.test_script] + passthrough_args + ["--seed", str(run_seed)]

        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.returncode != 0:
            print(f"--- ERROR: Test script failed to execute. ---")
            print(process.stderr)
            continue

        run_result = parse_mr_output(process.stdout)
        if run_result:
            run_result['run_number'] = i + 1
            all_runs_data.append(run_result)
            # Use the new metric_label for a clear printout
            print(
                f"  -> Run {i + 1} Result: {run_result['result']}, {run_result['metric_label']}: {run_result['change_pct']:.2f}%")

    if not all_runs_data:
        print("\nExperiment finished, but no successful runs were recorded.")
        return

    # --- AGGREGATE RESULTS ---
    df = pd.DataFrame(all_runs_data)
    num_passes = (df['result'] == 'PASSED').sum()
    pass_rate = (num_passes / len(df)) * 100 if len(df) > 0 else 0
    metric_type = df['metric_type'].iloc[0]

    summary_df = None
    raw_df = None

    if metric_type == 'reward':
        # --- REWARD-BASED SUMMARY (Matches original format) ---
        summary_data = {
            'Total Runs': len(df),
            'Passes': num_passes,
            'Fails': len(df) - num_passes,
            'Pass Rate (%)': pass_rate,
            'Avg. Reward Change (%)': df['change_pct'].mean(),
            'Std. Reward Change (%)': df['change_pct'].std(),
            'Avg. Baseline Reward': df['baseline_reward'].mean(),
            'Avg. Mutated Reward': df['mutated_reward'].mean()
        }
        summary_cols = ['Total Runs', 'Passes', 'Fails', 'Pass Rate (%)', 'Avg. Reward Change (%)', 'Std. Reward Change (%)', 'Avg. Baseline Reward', 'Avg. Mutated Reward']
        summary_df = pd.DataFrame([summary_data], columns=summary_cols)

        # --- REWARD-BASED RAW DATA (Matches original format) ---
        raw_df = df.rename(columns={'change_pct': 'reward_change_pct'})
        raw_cols = ['seed', 'baseline_reward', 'mutated_reward', 'reward_change_pct', 'result', 'run_number']
        raw_df = raw_df[raw_cols]

    elif metric_type == 'action':
        # --- ACTION-BASED SUMMARY (New, clear format) ---
        summary_data = {
            'Total Runs': len(df),
            'Passes': num_passes,
            'Fails': len(df) - num_passes,
            'Pass Rate (%)': pass_rate,
            'Avg. Divergence (%)': df['change_pct'].mean(),
            'Std. Divergence (%)': df['change_pct'].std(),
            'Avg. Total Steps': df['total_steps'].mean(),
            'Avg. Divergent Steps': df['divergent_steps'].mean()
        }
        summary_cols = ['Total Runs', 'Passes', 'Fails', 'Pass Rate (%)', 'Avg. Divergence (%)', 'Std. Divergence (%)', 'Avg. Total Steps', 'Avg. Divergent Steps']
        summary_df = pd.DataFrame([summary_data], columns=summary_cols)

        # --- ACTION-BASED RAW DATA (New, clear format) ---
        raw_df = df.rename(columns={'change_pct': 'divergence_pct'})
        raw_cols = ['seed', 'total_steps', 'divergent_steps', 'divergence_pct', 'result', 'run_number']
        raw_df = raw_df[raw_cols]

    else:
        print("Error: Unknown metric type found in results. Cannot generate summary.")
        return

    # --- DISPLAY SUMMARY ---
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(summary_df.to_string(index=False))
    print("-" * 50)

    # --- SAVE TO CSV FILE (IN 'Results' FOLDER) ---
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_str = "_".join(arg.replace('--', '') for arg in passthrough_args).replace('=', '_')
    base_filename = f"results_{args.test_script.replace('.py', '')}_{args_str}_{timestamp}.csv"

    output_filename = os.path.join(results_dir, base_filename)

    with open(output_filename, 'w') as f:
        f.write("SUMMARY\n")
        summary_df.to_csv(f, index=False, lineterminator='\n')
        f.write("\nRAW DATA\n")
        raw_df.to_csv(f, index=False, lineterminator='\n')

    print(f"Results saved to: {output_filename}")


if __name__ == "__main__":
    main()

'''
# ==================================================================================================
# --- REWARD-BASED METAMORPHIC TESTS (rew) ---
# These tests evaluate robustness by measuring the change in total reward.
# ==================================================================================================

# --- Brightness ---
# Increases or decreases the brightness of the game frames.
python run_experiment.py MR_brightness_rew.py --num_runs 50 --model ppo --threshold 15.0 --brightness_offset 30 
python run_experiment.py MR_brightness_rew.py --num_runs 50 --model dqn --threshold 15.0 --brightness_offset 30 
python run_experiment.py MR_brightness_rew.py --num_runs 50 --model ppo --threshold 15.0 --brightness_offset -30 
python run_experiment.py MR_brightness_rew.py --num_runs 50 --model dqn --threshold 15.0 --brightness_offset -30 

# --- Color Inversion ---
# Inverts the colors of the game frames.
python run_experiment.py MR_color_inversion_rew.py --num_runs 50 --model ppo --threshold 15.0
python run_experiment.py MR_color_inversion_rew.py --num_runs 50 --model dqn --threshold 15.0

# --- Horizontal Flip ---
# Flips the game frames horizontally.
python run_experiment.py MR_horizontal_flip_rew.py --num_runs 50 --model ppo --threshold 15.0
python run_experiment.py MR_horizontal_flip_rew.py --num_runs 50 --model dqn --threshold 15.0

# --- Noise Injection ---
# Adds random noise to the game frames.
python run_experiment.py MR_noise_injection_rew.py --num_runs 50 --model ppo --threshold 15.0 --noise_level 20.0 
python run_experiment.py MR_noise_injection_rew.py --num_runs 50 --model dqn --threshold 15.0 --noise_level 20.0 
python run_experiment.py MR_noise_injection_rew.py --num_runs 50 --model ppo --threshold 15.0 --noise_level 40.0 

# --- Random Occlusion ---
# Adds a black square to a random location in the game frames.
python run_experiment.py MR_random_occlusion_rew.py --num_runs 50 --model ppo --threshold 15.0 --occlusion_size 20 
python run_experiment.py MR_random_occlusion_rew.py --num_runs 50 --model dqn --threshold 15.0 --occlusion_size 20 
python run_experiment.py MR_random_occlusion_rew.py --num_runs 50 --model ppo --threshold 15.0 --occlusion_size 40 

# --- Score Blackout ---
# Blacks out the score area of the game frames.
python run_experiment.py MR_score_blackout_rew.py --num_runs 50 --model ppo --threshold 15.0 --score_rows 10
python run_experiment.py MR_score_blackout_rew.py --num_runs 50 --model dqn --threshold 15.0 --score_rows 10
python run_experiment.py MR_score_blackout_rew.py --num_runs 50 --model ppo --threshold 15.0 --score_rows 15 
python run_experiment.py MR_score_blackout_rew.py --num_runs 50 --model dqn --threshold 15.0 --score_rows 15 

# --- Action Drop (Probability) ---
# Drops agent actions (replaces with NOOP) with a given probability.
python run_experiment.py MR_action_drop_prob_rew.py --num_runs 50 --model ppo --threshold 15.0 --drop_probability 0.1
python run_experiment.py MR_action_drop_prob_rew.py --num_runs 50 --model dqn --threshold 15.0 --drop_probability 0.1

# --- Action Drop (Interval) ---
# Drops agent actions (replaces with NOOP) at a fixed interval.
python run_experiment.py MR_action_drop_interval_rew.py --num_runs 50 --model ppo --threshold 15.0 --drop_interval 10
python run_experiment.py MR_action_drop_interval_rew.py --num_runs 50 --model dqn --threshold 15.0 --drop_interval 10

# --- Action Noise (Probability) ---
# Replaces agent actions with random actions with a given probability.
python run_experiment.py MR_action_noise_prob_rew.py --num_runs 50 --model ppo --threshold 15.0 --noise_probability 0.1
python run_experiment.py MR_action_noise_prob_rew.py --num_runs 50 --model dqn --threshold 15.0 --noise_probability 0.1

# --- Action Noise (Interval) ---
# Replaces agent actions with random actions at a fixed interval.
python run_experiment.py MR_action_noise_interval_rew.py --num_runs 50 --model ppo --threshold 15.0 --noise_interval 10
python run_experiment.py MR_action_noise_interval_rew.py --num_runs 50 --model dqn --threshold 15.0 --noise_interval 10

# --- Frame Drop (Probability) ---
# Skips frames with a given probability.
python run_experiment.py MR_frame_drop_prob_rew.py --num_runs 50 --model ppo --threshold 15.0 --drop_probability 0.5
python run_experiment.py MR_frame_drop_prob_rew.py --num_runs 50 --model dqn --threshold 15.0 --drop_probability 0.5

# --- Frame Drop (Interval) ---
# Skips frames at a fixed interval.
python run_experiment.py MR_frame_drop_interval_rew.py --num_runs 50 --model ppo --threshold 15.0 --drop_interval 2
python run_experiment.py MR_frame_drop_interval_rew.py --num_runs 50 --model dqn --threshold 15.0 --drop_interval 2

# --- Frame Stutter (Probability) ---
# Repeats the last frame with a given probability.
python run_experiment.py MR_frame_stutter_prob_rew.py --num_runs 50 --model ppo --threshold 15.0 --stutter_probability 0.5
python run_experiment.py MR_frame_stutter_prob_rew.py --num_runs 50 --model dqn --threshold 15.0 --stutter_probability 0.5

# --- Frame Stutter (Interval) ---
# Repeats the last frame at a fixed interval.
python run_experiment.py MR_frame_stutter_interval_rew.py --num_runs 50 --model ppo --threshold 15.0 --stutter_interval 2
python run_experiment.py MR_frame_stutter_interval_rew.py --num_runs 50 --model dqn --threshold 15.0 --stutter_interval 2

# --- Frame Lag ---
# Delays actions by a fixed number of frames.
python run_experiment.py MR_frame_lag_rew.py --num_runs 50 --model ppo --threshold 15.0 --lag_amount 2
python run_experiment.py MR_frame_lag_rew.py --num_runs 50 --model dqn --threshold 15.0 --lag_amount 2


# ==================================================================================================
# --- ACTION-BASED METAMORPHIC TESTS (act) ---
# These tests evaluate robustness by measuring the divergence in agent actions.
# ==================================================================================================

# --- Brightness ---
# Increases or decreases the brightness of the game frames.
python run_experiment.py MR_brightness_act.py --num_runs 50 --model ppo --threshold 15.0 --brightness_offset 10
python run_experiment.py MR_brightness_act.py --num_runs 50 --model dqn --threshold 15.0 --brightness_offset 10
python run_experiment.py MR_brightness_act.py --num_runs 50 --model ppo --threshold 15.0 --brightness_offset 30
python run_experiment.py MR_brightness_act.py --num_runs 50 --model dqn --threshold 15.0 --brightness_offset 30
python run_experiment.py MR_brightness_act.py --num_runs 50 --model ppo --threshold 15.0 --brightness_offset -10
python run_experiment.py MR_brightness_act.py --num_runs 50 --model dqn --threshold 15.0 --brightness_offset -10
python run_experiment.py MR_brightness_act.py --num_runs 50 --model ppo --threshold 15.0 --brightness_offset -30
python run_experiment.py MR_brightness_act.py --num_runs 50 --model dqn --threshold 15.0 --brightness_offset -30

# --- Horizontal Flip ---
# Flips the game frames horizontally.
python run_experiment.py MR_horizontal_flip_act.py --num_runs 50 --model ppo --threshold 15.0
python run_experiment.py MR_horizontal_flip_act.py --num_runs 50 --model dqn --threshold 15.0

# --- Color Inversion ---
# Inverts the colors of the game frames.
python run_experiment.py MR_color_inversion_act.py --num_runs 50 --model ppo --threshold 15.0
python run_experiment.py MR_color_inversion_act.py --num_runs 50 --model dqn --threshold 15.0

# --- Score Blackout ---
# Blacks out the score area of the game frames.
python run_experiment.py MR_score_blackout_act.py --num_runs 50 --model ppo --threshold 15.0 --score_rows 10
python run_experiment.py MR_score_blackout_act.py --num_runs 50 --model dqn --threshold 15.0 --score_rows 10
python run_experiment.py MR_score_blackout_act.py --num_runs 50 --model ppo --threshold 15.0 --score_rows 15 
python run_experiment.py MR_score_blackout_act.py --num_runs 50 --model dqn --threshold 15.0 --score_rows 15

# --- Noise Injection ---
# Adds random noise to the game frames.
python run_experiment.py MR_noise_injection_act.py --num_runs 50 --model ppo --threshold 15.0 --noise_level 20.0
python run_experiment.py MR_noise_injection_act.py --num_runs 50 --model dqn --threshold 15.0 --noise_level 20.0
python run_experiment.py MR_noise_injection_act.py --num_runs 50 --model ppo --threshold 15.0 --noise_level 40.0

# --- Random Occlusion ---
# Adds a black square to a random location in the game frames.
python run_experiment.py MR_random_occlusion_act.py --num_runs 50 --model ppo --threshold 15.0 --occlusion_size 20
python run_experiment.py MR_random_occlusion_act.py --num_runs 50 --model dqn --threshold 15.0 --occlusion_size 20
python run_experiment.py MR_random_occlusion_act.py --num_runs 50 --model ppo --threshold 15.0 --occlusion_size 40


# ==================================================================================================
# --- SPECIALIZED/LONG-DURATION RUNS ---
# ==================================================================================================

# Example of a longer run with a specific master seed for reproducibility.
python run_experiment.py MR_horizontal_flip_rew.py --num_runs 200 --model ppo --threshold 15.0 --master_seed 47690
python run_experiment.py MR_horizontal_flip_rew.py --num_runs 200 --model dqn --threshold 15.0 --master_seed 47690
python run_experiment.py MR_horizontal_flip_act.py --num_runs 200 --model ppo --threshold 15.0 --master_seed 47690
python run_experiment.py MR_horizontal_flip_act.py --num_runs 200 --model dqn --threshold 15.0 --master_seed 47690

# Example of a longer run with a specific master seed for reproducibility.
python run_experiment.py MR_horizontal_flip_rew.py --num_runs 1000 --model ppo --threshold 15.0 --master_seed 17428
python run_experiment.py MR_horizontal_flip_rew.py --num_runs 1000 --model dqn --threshold 15.0 --master_seed 17428
python run_experiment.py MR_horizontal_flip_act.py --num_runs 1000 --model ppo --threshold 15.0 --master_seed 17428
python run_experiment.py MR_horizontal_flip_act.py --num_runs 1000 --model dqn --threshold 15.0 --master_seed 17428
'''