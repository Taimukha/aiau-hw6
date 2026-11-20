
import csv
import random
from pathlib import Path
from statistics import mean

# ==========================================
# CONFIGURATION
# ==========================================

GAMMA = 0.9
TOLERANCE = 1e-6
MAX_ITERATIONS = 10000

BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
MAX_STEPS = 100

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

ORIGINAL_MAP = [
    ".M.G",
    ".#IH",
    "S..."
]

EXTENDED_MAP = [
    ".M.G",
    ".#IH",
    "SD.W"
]

REWARDS = {
    ".": -1,
    "S": -1,
    "M": -3,
    "I": -1,
    "D": -2,
    "W": -5,
    "G": 10,
    "H": -10
}

# Tie-breaking priority:
# UP -> RIGHT -> DOWN -> LEFT
ACTIONS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1)
}

ARROWS = {
    "UP": "↑",
    "RIGHT": "→",
    "DOWN": "↓",
    "LEFT": "←"
}


# ==========================================
# GRID WORLD RULES
# ==========================================

def get_states(grid):
    states = []

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != "#":
                states.append((r, c))

    return states


def is_terminal(grid, state):
    r, c = state
    return grid[r][c] in ("G", "H")


def move(grid, state, action):
    dr, dc = ACTIONS[action]
    r, c = state

    nr = r + dr
    nc = c + dc

    if (
        nr < 0
        or nr >= len(grid)
        or nc < 0
        or nc >= len(grid[nr])
        or grid[nr][nc] == "#"
    ):
        return state

    return (nr, nc)


def get_transitions(grid, state, action):
    if is_terminal(grid, state):
        return []

    r, c = state

    if grid[r][c] == "I":
        perpendicular = {
            "UP": ["LEFT", "RIGHT"],
            "DOWN": ["LEFT", "RIGHT"],
            "LEFT": ["UP", "DOWN"],
            "RIGHT": ["UP", "DOWN"]
        }

        possible_actions = [
            (action, 0.8),
            (perpendicular[action][0], 0.1),
            (perpendicular[action][1], 0.1)
        ]
    else:
        possible_actions = [(action, 1.0)]

    transitions = []

    for actual_action, probability in possible_actions:
        next_state = move(grid, state, actual_action)
        nr, nc = next_state
        reward = REWARDS[grid[nr][nc]]

        transitions.append(
            (probability, next_state, reward)
        )

    return transitions


# ==========================================
# VALUE ITERATION
# ==========================================

def action_value(grid, state, action, values):
    total = 0.0

    for probability, next_state, reward in get_transitions(
        grid, state, action
    ):
        total += probability * (
            reward + GAMMA * values[next_state]
        )

    return total


def value_iteration(grid):
    states = get_states(grid)
    values = {state: 0.0 for state in states}
    history = [values.copy()]

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = values.copy()

        for state in states:
            if is_terminal(grid, state):
                new_values[state] = 0.0
                continue

            action_values = [
                action_value(grid, state, action, values)
                for action in ACTIONS
            ]

            new_values[state] = max(action_values)

        delta = max(
            abs(new_values[state] - values[state])
            for state in states
        )

        values = new_values
        history.append(values.copy())

        if delta < TOLERANCE:
            return values, history, iteration, delta

    raise RuntimeError("Value iteration did not converge")


def get_policy(grid, values):
    policy = {}

    for state in get_states(grid):
        if is_terminal(grid, state):
            continue

        best_action = max(
            ACTIONS,
            key=lambda action: action_value(
                grid, state, action, values
            )
        )

        policy[state] = best_action

    return policy


# ==========================================
# PRINT MAPS
# ==========================================

def print_values(grid, values):
    for r, row in enumerate(grid):
        cells = []

        for c, tile in enumerate(row):
            if tile == "#":
                cells.append("  ##### ")
            elif tile in ("G", "H"):
                cells.append(f"  {tile:^5} ")
            else:
                cells.append(f"{values[(r, c)]:7.3f} ")

        print(" ".join(cells))


def print_policy(grid, policy):
    for r, row in enumerate(grid):
        cells = []

        for c, tile in enumerate(row):
            if tile in ("#", "G", "H"):
                cells.append(tile)
            else:
                cells.append(ARROWS[policy[(r, c)]])

        print(" ".join(cells))


def save_map(name, grid, values, policy, iterations, delta):
    path = OUTPUT / f"part_c_{name}_map.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Map: {name}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Final delta: {delta}\n\n")

        f.write("Values:\n")
        for r, row in enumerate(grid):
            line = []

            for c, tile in enumerate(row):
                if tile in ("G", "H", "#"):
                    line.append(f"{tile:>8}")
                else:
                    line.append(f"{values[(r, c)]:8.3f}")

            f.write(" ".join(line) + "\n")

        f.write("\nPolicy:\n")
        for r, row in enumerate(grid):
            line = []

            for c, tile in enumerate(row):
                if tile in ("G", "H", "#"):
                    line.append(tile)
                else:
                    line.append(ARROWS[policy[(r, c)]])

            f.write(" ".join(line) + "\n")


# ==========================================
# SIMULATION
# ==========================================

def simulate_episode(grid, policy, episode_seed):
    rng = random.Random(episode_seed)

    start = next(
        (r, c)
        for r, row in enumerate(grid)
        for c, tile in enumerate(row)
        if tile == "S"
    )

    state = start
    discounted_return = 0.0

    for t in range(MAX_STEPS):
        action = policy[state]
        transitions = get_transitions(grid, state, action)

        selected = rng.choices(
            transitions,
            weights=[item[0] for item in transitions],
            k=1
        )[0]

        probability, next_state, reward = selected

        discounted_return += (GAMMA ** t) * reward
        state = next_state

        if is_terminal(grid, state):
            outcome = (
                "goal"
                if grid[state[0]][state[1]] == "G"
                else "hole"
            )

            return t + 1, discounted_return, outcome

    return MAX_STEPS, discounted_return, "time limit"


# ==========================================
# RULE CHECKS
# ==========================================

def run_rule_checks():
    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))

    grid = ORIGINAL_MAP
    extended = EXTENDED_MAP

    check(
        "Top boundary stays in place",
        move(grid, (0, 0), "UP") == (0, 0)
    )

    check(
        "Left boundary stays in place",
        move(grid, (0, 0), "LEFT") == (0, 0)
    )

    check(
        "Wall blocks movement",
        move(grid, (1, 0), "RIGHT") == (1, 0)
    )

    ice = get_transitions(grid, (1, 2), "UP")

    check(
        "Ice probabilities sum to 1",
        abs(sum(p for p, _, _ in ice) - 1.0) < 1e-12
    )

    check(
        "Ice probabilities are 0.8, 0.1, 0.1",
        sorted(round(p, 10) for p, _, _ in ice)
        == [0.1, 0.1, 0.8]
    )

    check(
        "Mud reward is -3",
        get_transitions(grid, (0, 0), "RIGHT")[0][2] == -3
    )

    check(
        "Sand reward is -2",
        get_transitions(extended, (2, 0), "RIGHT")[0][2] == -2
    )

    check(
        "Water reward is -5",
        get_transitions(extended, (2, 2), "RIGHT")[0][2] == -5
    )

    check(
        "Goal reward is +10",
        get_transitions(grid, (0, 2), "RIGHT")[0][2] == 10
    )

    # FIXED: Check the transition from Ice to Hole.
    check(
        "Hole reward is -10",
        any(
            next_state == (1, 3) and reward == -10
            for probability, next_state, reward
            in get_transitions(grid, (1, 2), "RIGHT")
        )
    )

    check(
        "Goal has no outgoing transitions",
        get_transitions(grid, (0, 3), "LEFT") == []
    )

    check(
        "Hole has no outgoing transitions",
        get_transitions(grid, (1, 3), "LEFT") == []
    )

    check(
        "Blocked move uses current tile reward",
        get_transitions(grid, (0, 1), "DOWN")[0][2] == -3
    )

    with open(
        OUTPUT / "part_c_rule_checks.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            f.write(f"{status}: {name}\n")
            print(f"{status}: {name}")

    assert all(
        passed for _, passed in checks
    ), "Rule check failed"


# ==========================================
# RUN EXPERIMENTS
# ==========================================

def run_experiments():
    all_rows = []
    summary_rows = []

    maps = {
        "original": ORIGINAL_MAP,
        "extended": EXTENDED_MAP
    }

    for name, grid in maps.items():
        print(f"\n===== {name.upper()} MAP =====")

        values, history, iterations, delta = value_iteration(grid)
        policy = get_policy(grid, values)

        print("\nV1:")
        print_values(grid, history[1])

        print("\nV2:")
        print_values(grid, history[2])

        print("\nFinal values:")
        print_values(grid, values)

        print("\nOptimal policy:")
        print_policy(grid, policy)

        print("\nIterations:", iterations)
        print("Final delta:", delta)

        save_map(name, grid, values, policy, iterations, delta)

        print("\nSimulation results:")

        for batch_seed in BATCH_SEEDS:
            batch_rows = []

            for i in range(EPISODES_PER_BATCH):
                episode_seed = 1000 * batch_seed + i

                steps, discounted_return, outcome = simulate_episode(
                    grid,
                    policy,
                    episode_seed
                )

                row = {
                    "map": name,
                    "batch_seed": batch_seed,
                    "episode_seed": episode_seed,
                    "steps": steps,
                    "discounted_return": discounted_return,
                    "outcome": outcome
                }

                all_rows.append(row)
                batch_rows.append(row)

            goals = sum(
                r["outcome"] == "goal" for r in batch_rows
            )

            holes = sum(
                r["outcome"] == "hole" for r in batch_rows
            )

            limits = sum(
                r["outcome"] == "time limit" for r in batch_rows
            )

            summary = {
                "map": name,
                "batch_seed": batch_seed,
                "goals": goals,
                "holes": holes,
                "time_limits": limits,
                "mean_steps": round(
                    mean(r["steps"] for r in batch_rows), 3
                ),
                "mean_discounted_return": round(
                    mean(
                        r["discounted_return"]
                        for r in batch_rows
                    ),
                    6
                )
            }

            summary_rows.append(summary)

            print(
                f"Batch={batch_seed}: "
                f"Goals={goals}, "
                f"Holes={holes}, "
                f"Limits={limits}, "
                f"Mean steps={summary['mean_steps']}, "
                f"Mean discounted return="
                f"{summary['mean_discounted_return']}"
            )

    def save_csv(filename, rows):
        with open(
            OUTPUT / filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    save_csv("part_c_episodes.csv", all_rows)
    save_csv("part_c_summary.csv", summary_rows)

    print("\nTotal Part C episodes:", len(all_rows))


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("===== PART C RULE CHECKS =====")
    run_rule_checks()

    print("\n===== VALUE ITERATION AND SIMULATION =====")
    run_experiments()

    print("\nAll Part C results saved in outputs/")
