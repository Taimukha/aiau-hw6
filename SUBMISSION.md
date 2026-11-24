# HW6 Submission

**Course:** CSS-4007 — Artificial Intelligence  
**Assignment:** HW6 — Grid World: Calculate, Simulate, Solve

## AI-use declaration

I used ChatGPT (OpenAI) to help understand the assignment requirements, explain MDPs and value iteration, check paper calculations, develop Python code for the FrozenLake experiments, and organize the submission report.

I ran the Python experiments locally, collected the results, and compared the simulation outcomes. AI assistance was used for explanations, coding support, and documentation.

## Setup and reproduction

**Environment:**
- Operating system: macOS
- Python: 3.9 (exact patch version to be confirmed)
- Gymnasium: 1.1.1
- NumPy: 2.0.2

**Installation commands:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install gymnasium
```

**Check installed versions:**

```bash
python --version
python -m pip show gymnasium numpy
```

**Run Part B:**

```bash
python part_b.py
```

**Run Part C:**

```bash
python part_c.py
```

The Part C command will be available after implementing the custom solver.

## Part A — Paper calculations

**Paper calculation scan:**

[View Part A paper calculations](HW6%20-%20ParsA%20Paper%20Colculation%20s.pdf)

### 1. States, actions, transitions and rewards

**States:** All positions on the grid except the wall. The goal and hole are terminal states.

**Actions:** The agent can choose Up, Right, Down, or Left.

**Transitions:** Movement is deterministic on ordinary floor, start, and mud. On ice, the intended direction has probability 0.8, while each perpendicular direction has probability 0.1. Blocked moves leave the agent in the same position.

**Rewards:** Rewards depend on the destination tile. Ordinary floor and start give −1, mud gives −3, goal gives +10, and hole gives −10.

**Objective:** The agent tries to maximize the expected total discounted reward.

The discount factor is 0.9, and the initial values are zero.

### 2. V1 calculations

Starting from V0 = 0, the first value iteration considers the immediate expected rewards.

Examples:

```text
V1(1,3) = 10 + 0.9(0) = 10
V1(1,2) = -1 + 0.9(0) = -1
V1(2,3) = -1
```

**V1 map:**

```text
        Col 1   Col 2   Col 3   Col 4

Row 1    -1      -1      10      G(0)
Row 2    -1       #      -1      H(0)
Row 3    -1      -1      -1       -1
```

The goal and hole have terminal value zero. The wall is not a state.

### 3. V2 calculations

The value iteration formula is:

```text
V(k+1,s) = max over actions of
sum P(s'|s,a) × [R(s,a,s') + 0.9 × V(k,s')]
```

**Position (1,2) — Mud**

```text
Up:    -3 + 0.9(-1) = -3.9
Right: -1 + 0.9(10) = 8
Down:  -3 + 0.9(-1) = -3.9
Left:  -1 + 0.9(-1) = -1.9

V2(1,2) = 8
Best action: Right
```

**Position (1,3) — Ordinary floor**

```text
Up:    -1 + 0.9(10) = 8
Right: 10 + 0.9(0) = 10
Down:  -1 + 0.9(-1) = -1.9
Left:  -3 + 0.9(-1) = -3.9

V2(1,3) = 10
Best action: Right
```

**Position (2,3) — Ice**

```text
Up:
0.8(8) + 0.1(-1.9) + 0.1(-10)
= 5.21

Right:
0.8(-10) + 0.1(8) + 0.1(-1.9)
= -7.39

Down:
0.8(-1.9) + 0.1(-1.9) + 0.1(-10)
= -2.71

Left:
0.8(-1.9) + 0.1(8) + 0.1(-1.9)
= -0.91

V2(2,3) = 5.21
Best action: Up
```

### 4. Best actions and ties

| Position | V2 | Best action | Ties |
|---|---:|---|---|
| (1,2) | 8 | Right | None |
| (1,3) | 10 | Right | None |
| (2,3) | 5.21 | Up | None |

**Best-action map:**

```text
.  →  →  G
.  #  ↑  H
S  .  .  .
```

Two iterations are not sufficient to prove the final optimal policy because they consider only a limited number of future rewards. More iterations are needed to evaluate long-term consequences.

## Part B — FrozenLake

### Fixed policy, written before the experiments

I designed a fixed policy based on the 4×4 FrozenLake map.

The intended path is:

```text
0 → 4 → 8 → 9 → 13 → 14 → 15
```

This route avoids holes when movement is deterministic.

**Fixed policy:**

| State | Action |
|---|---|
| 0 | Down |
| 1 | Right |
| 2 | Down |
| 3 | Left |
| 4 | Down |
| 6 | Down |
| 8 | Right |
| 9 | Down |
| 10 | Down |
| 13 | Right |
| 14 | Right |

The policy was kept unchanged throughout all experiments.

The random policy selected each action uniformly using a seeded random generator.


2. Fixed Policy және Random Policy
Мен екі стратегияны қолдандым.

Біріншісі — Fixed Policy.

Бұл алдын ала жазылған тұрақты стратегия. Мен роботқа қауіпсіз маршрут таңдадым:
0 → 4 → 8 → 9 → 13 → 14 → 15


Эксперимент барысында бұл маршрутты өзгертпедім.

Екіншісі — Random Policy.

Бұл стратегияда агент бағытты кездейсоқ таңдайды. Мысалы, жоғары, төмен, оңға немесе солға жүруі мүмкін.
3. Slippery және Non-slippery
Мен екі стратегияны екі түрлі ортада тексердім.
- Slippery=False — агент таңдаған бағытпен нақты жүреді.
- Slippery=True — агент тайып, басқа бағытқа ауытқуы мүмкін.
Сондықтан Slippery=True кезінде Goal-ға жету қиындайды.
4. Эксперимент қалай жүргізілді?
Мен төрт түрлі жағдайды тексердім:
1. Random Policy — тайғанамайтын ортада.
2. Fixed Policy — тайғанамайтын ортада.
3. Random Policy — тайғанақ ортада.
4. Fixed Policy — тайғанақ ортада.
Әр жағдайға 150 ойыннан өткіздім. Ол үш Batch-тан тұрды, әр Batch-та 50 ойын болды.

Жалпы 600 ойын орындалды.

Әр ойын ең көбі 100 қадамға созылды.
5. Эксперимент нәтижелері
Нәтижелерге қарасақ:
- Random Policy тайғанамайтын ортада 0.67% ғана нәтиже көрсетті.
- Fixed Policy тайғанамайтын ортада 100% нәтижеге жетті.
- Random Policy тайғанақ ортада 0% көрсетті.
- Fixed Policy тайғанақ ортада 2.67% нәтиже көрсетті.
Неге бұлай болды?

Fixed Policy тайғанамайтын ортада жақсы жұмыс істеді, себебі агент алдын ала белгіленген қауіпсіз маршрутпен жүрді.

Ал тайғанақ ортада агент басқа бағытқа ауытқып, тесікке түсіп қалуы мүмкін. Сондықтан оның нәтижесі төмендеді.

Random Policy нашар нәтиже көрсетті, себебі агент бағытты кездейсоқ таңдады және Goal-ға жетудің нақты стратегиясы болмады.
6. Қорытынды
Қорытындылай келе, Fixed Policy тайғанамайтын ортада ең жақсы нәтиже көрсетті.

Бірақ Slippery=True болғанда оның тиімділігі төмендеді. Бұл қозғалыстың кездейсоқ өзгеруі агенттің нәтижесіне қатты әсер ететінін көрсетті.

Бұл тапсырмада мен агентті үйреткен жоқпын. Тек екі дайын стратегияны салыстырып, олардың нәтижелерін зерттедім.

### Experimental setup

I used `FrozenLake-v1` with the default 4×4 map and rewards.

I compared four conditions:

1. Random policy, non-slippery environment.
2. Fixed policy, non-slippery environment.
3. Random policy, slippery environment.
4. Fixed policy, slippery environment.

Each condition was tested using three batches of 50 episodes.

Batch seeds were 11, 22, and 33.

Episode seeds followed:

```text
episode_seed = 1000 * batch_seed + i
```

Each episode had a maximum of 100 steps.

Total episodes: **600**.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
|---|---|---|---|---|---|---|---|
| False | Random | 11 | 0 | 50 | 0 | 0% | 8.14 |
| False | Random | 22 | 1 | 49 | 0 | 2% | 7.56 |
| False | Random | 33 | 0 | 50 | 0 | 0% | 6.88 |
| False | Fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| False | Fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| False | Fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| True | Random | 11 | 0 | 50 | 0 | 0% | 8.06 |
| True | Random | 22 | 0 | 50 | 0 | 0% | 7.60 |
| True | Random | 33 | 0 | 50 | 0 | 0% | 7.98 |
| True | Fixed | 11 | 1 | 49 | 0 | 2% | 4.24 |
| True | Fixed | 22 | 2 | 48 | 0 | 4% | 5.48 |
| True | Fixed | 33 | 1 | 49 | 0 | 2% | 4.44 |

### Overall results

| Environment | Policy | Goals /150 | Success rate |
|---|---|---:|---:|
| Non-slippery | Random | 1 | 0.67% |
| Non-slippery | Fixed | 150 | 100% |
| Slippery | Random | 0 | 0% |
| Slippery | Fixed | 4 | 2.67% |

### Explanation of results

**Effect of slipping**

Without slipping, the fixed policy reached the goal in all 150 episodes. When slipping was enabled, its success rate decreased to 2.67%.

This happened because the agent could move in an unintended direction and fall into a hole.

The random policy performed poorly in both environments because its actions were not selected to reach the goal.

**Consistency across batches**

The non-slippery fixed policy achieved identical results in all three batches.

The slippery fixed policy achieved success rates of 2%, 4%, and 2%, showing small variation between batches.

The random policy had consistently low success rates.

**Why one successful episode is insufficient**

One successful episode can happen by chance. Multiple episodes and batches provide more reliable evidence of how well a policy performs.

**Why these experiments are not training**

The policies were fixed before the experiments and were not updated based on rewards or experience.

The experiments evaluated existing policies rather than learning new ones.

### Saved results and traces

- [All 600 episode results](outputs/part_b_episodes.csv)
- [Batch summaries](outputs/part_b_summary.csv)
- [Successful and failed episode traces](outputs/part_b_traces.txt)

## Part C — Custom Solver

### Compare code with paper

I implemented value iteration manually using the same Grid World rules as Part A.

The solver uses a discount factor of 0.9 and initializes all state values to zero.

**V1 — Original map**

```text
-1.000  -1.000  10.000    G
-1.000     #    -1.000    H
-1.000  -1.000  -1.000  -1.000
```

**V2 — Original map**

```text
-1.900   8.000  10.000    G
-1.900     #     5.210    H
-1.900  -1.900  -1.900  -1.900
```

**Comparison with paper calculations**

| Position | Paper V2 | Code V2 | Difference |
|---|---:|---:|---:|
| (1,2) Mud | 8.000 | 8.000 | 0 |
| (1,3) Floor | 10.000 | 10.000 | 0 |
| (2,3) Ice | 5.210 | 5.210 | 0 |

All three requested V2 values match the paper calculations. No discrepancies were found in these values.

### Values, policies and convergence

**Original map — final values**

```text
4.200   8.000  10.000    G
2.780     #     5.824    H
1.536   2.818   4.242   2.818
```

**Original map — optimal policy**

```text
→ → → G
↑ # ↑ H
→ → ↑ ←
```

**Extended map — final values**

```text
4.200   8.000  10.000    G
2.780     #     5.824    H
1.502   2.818   4.242   2.818
```

**Extended map — optimal policy**

```text
→ → → G
↑ # ↑ H
↑ → ↑ ←
```

**Convergence results**

| Map | Iterations | Final maximum change | Converged |
|---|---:|---:|---|
| Original | 12 | 2.1652931048166124e-7 | Yes |
| Extended | 11 | 2.405881232014906e-7 | Yes |

Both maps converged with a tolerance of 1e-6 and a maximum iteration limit of 10,000.

For equal action values, the tie-breaking priority was Up, Right, Down, Left.

**Saved maps**

- [Original map values and policy](outputs/part_c_original_map.txt)
- [Extended map values and policy](outputs/part_c_extended_map.txt)

### Simulation results

I simulated the computed optimal policy on each map using three batches of 50 episodes. Each episode had a maximum of 100 steps.

The episode seed formula was `1000 * batch_seed + i`.

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
|---|---|---:|---:|---:|---:|---:|
| Original | 11 | 42 | 8 | 0 | 4.98 | 0.891260 |
| Original | 22 | 46 | 4 | 0 | 5.04 | 1.914776 |
| Original | 33 | 48 | 2 | 0 | 5.04 | 2.492144 |
| Extended | 11 | 50 | 0 | 0 | 5.00 | 1.502000 |
| Extended | 22 | 50 | 0 | 5.00 | 1.502000 |
| Extended | 33 | 50 | 0 | 5.00 | 1.502000 |

**Overall simulation results**

The original map achieved 136 goals out of 150 episodes, corresponding to a success rate of 90.67%.

The extended map achieved 150 goals out of 150 episodes, corresponding to a success rate of 100%.

No episodes reached the time limit.

**Saved results**

- [All 300 episode results](outputs/part_c_episodes.csv)
- [Simulation summary](outputs/part_c_summary.csv)

### Explain your results

**Which values and actions changed with sand and water?**

In the extended map, sand (D) has a reward of −2 and water (W) has a reward of −5 when entered.

The value of the starting state changed from 1.536 to 1.502. The optimal action at the starting state changed from Right to Up.

The other displayed state values remained the same to three decimal places.

**Where did cost and risk matter?**

In the original map, the optimal policy travels through the ice tile. This route can lead to the goal but carries a risk of slipping into the hole.

In the extended map, the additional cost of sand changes the preferred route. The optimal policy follows a safe path that avoids ice and water.

**How do simulations relate to computed values?**

Value iteration calculates the expected discounted return under the model's transition probabilities over an infinite horizon.

The simulations estimate discounted returns using sampled episodes with a maximum of 100 steps.

The original map has stochastic ice transitions, so individual batches produce different average returns. Its observed batch means were 0.891260, 1.914776, and 2.492144.

The extended map's optimal path is deterministic. All 150 episodes reached the goal in five steps, producing a discounted return of 1.502 in every batch.

The simulation results are consistent with the computed policies. Differences between sampled returns and expected values can occur because of randomness and the finite simulation horizon.

### Rule checks

I implemented 13 rule checks to verify the Grid World environment.

| Rule | Result |
|---|---|
| Top boundary stays in place | PASS |
| Left boundary stays in place | PASS |
| Wall blocks movement | PASS |
| Ice probabilities sum to 1 | PASS |
| Ice probabilities are 0.8, 0.1, 0.1 | PASS |
| Mud reward is −3 | PASS |
| Sand reward is −2 | PASS |
| Water reward is −5 | PASS |
| Goal reward is +10 | PASS |
| Hole reward is −10 | PASS |
| Goal has no outgoing transitions | PASS |
| Hole has no outgoing transitions | PASS |
| Blocked move uses current tile reward | PASS |

All 13 checks passed.

[View rule checks](outputs/part_c_rule_checks.txt)

## Conclusion

In Part A, I calculated the first value iteration updates manually.

In Part B, I compared random and fixed policies using 600 FrozenLake episodes under deterministic and slippery conditions.

In Part C, I implemented a custom value iteration solver, compared its calculations with Part A, and evaluated optimal policies using 300 simulated episodes.

The experiments demonstrate how transition uncertainty, movement costs, and reward structure influence optimal decisions.

The results also show the difference between computing expected values and observing returns from individual simulations.
