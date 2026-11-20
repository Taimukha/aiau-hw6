
import csv
import random
from pathlib import Path
from statistics import mean

import gymnasium as gym

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
MAX_STEPS = 100

# FrozenLake actions:
# 0 = Left, 1 = Down, 2 = Right, 3 = Up

# Fixed policy designed before experiments.
# One action for each nonterminal state.
FIXED_POLICY = {
    0: 1,   # Down
    1: 2,   # Right
    2: 1,   # Down
    3: 0,   # Left
    4: 1,   # Down
    6: 1,   # Down
    8: 2,   # Right
    9: 1,   # Down
    10: 1,  # Down
    13: 2,  # Right
    14: 2,  # Right
}

GOAL_STATE = 15
HOLE_STATES = {5, 7, 11, 12}


def run_episode(env, policy, episode_seed, action_seed,
                save_trace=False):
    state, info = env.reset(seed=episode_seed)
    rng = random.Random(action_seed)

    total_reward = 0
    steps = 0
    outcome = "time limit"
    trace = []

    if save_trace:
        trace.append("Initial map:\n" + env.render())

    for step in range(MAX_STEPS):
        if policy == "random":
            action = rng.randrange(4)
        else:
            action = FIXED_POLICY[state]

        state, reward, terminated, truncated, info = env.step(action)

        steps = step + 1
        total_reward += reward

        if save_trace:
            trace.append(
                f"Step {steps}, action={action}, "
                f"state={state}, reward={reward}\n"
                + env.render()
            )

        # Check termination before the time limit.
        if terminated:
            if state == GOAL_STATE:
                outcome = "goal"
            elif state in HOLE_STATES:
                outcome = "hole"
            else:
                raise ValueError(f"Unexpected terminal state: {state}")
            break

        if truncated:
            outcome = "time limit"
            break

    return steps, total_reward, outcome, trace


all_rows = []
summary_rows = []
example_traces = {}

for slippery in [False, True]:
    for policy in ["random", "fixed"]:
        for batch_seed in BATCH_SEEDS:

            env = gym.make(
                "FrozenLake-v1",
                map_name="4x4",
                is_slippery=slippery,
                render_mode="ansi",
                max_episode_steps=MAX_STEPS
            )

            batch_rows = []

            for i in range(EPISODES):
                episode_seed = 1000 * batch_seed + i

                steps, reward, outcome, trace = run_episode(
                    env,
                    policy,
                    episode_seed,
                    action_seed=episode_seed,
                    save_trace=True
                )

                row = {
                    "condition": f"slippery={slippery}",
                    "batch_seed": batch_seed,
                    "episode_seed": episode_seed,
                    "policy": policy,
                    "steps": steps,
                    "total_reward": reward,
                    "outcome": outcome
                }

                all_rows.append(row)
                batch_rows.append(row)

                if outcome in ["goal", "hole"] and outcome not in example_traces:
                    example_traces[outcome] = (
                        f"Outcome: {outcome}\n"
                        f"Slippery: {slippery}\n"
                        f"Policy: {policy}\n"
                        f"Episode seed: {episode_seed}\n\n"
                        + "\n".join(trace)
                    )

            goals = sum(r["outcome"] == "goal" for r in batch_rows)
            holes = sum(r["outcome"] == "hole" for r in batch_rows)
            limits = sum(r["outcome"] == "time limit" for r in batch_rows)

            summary = {
                "slippery": slippery,
                "policy": policy,
                "batch_seed": batch_seed,
                "goals": goals,
                "holes": holes,
                "time_limits": limits,
                "success_percent": goals / EPISODES * 100,
                "mean_steps": round(
                    mean(r["steps"] for r in batch_rows), 2
                )
            }

            summary_rows.append(summary)

            print(
                f"Slippery={slippery}, Policy={policy}, "
                f"Batch={batch_seed}: "
                f"Goals={goals}, Holes={holes}, "
                f"Limits={limits}, "
                f"Success={summary['success_percent']}%, "
                f"Mean steps={summary['mean_steps']}"
            )

            env.close()


def save_csv(filename, rows):
    with open(OUTPUT / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


save_csv("part_b_episodes.csv", all_rows)
save_csv("part_b_summary.csv", summary_rows)

with open(OUTPUT / "part_b_traces.txt", "w") as f:
    for outcome in ["goal", "hole"]:
        if outcome in example_traces:
            f.write(example_traces[outcome] + "\n\n")
        else:
            f.write(f"No {outcome} episode observed.\n")

print("\nTotal episodes:", len(all_rows))
print("Results saved in outputs/")
