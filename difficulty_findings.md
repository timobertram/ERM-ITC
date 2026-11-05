# Instance difficulty for CB-CTT: current findings

*2026-11-03. Tool: `dataset/ctt_difficulty.py`: 8 structural features, CP-SAT/MILP runs and rank-correlation analysis. `--nn` adds 11 network-specific measures; `--analyze CSV --instance-log LOG...` relates the measures to per-instance network results logged by `pretrain_ctt.py` (`training.instance_log`). Result files: solver runs in `data/ITC2007/synth/difficulty_runs/`, network experiments in `data/ITC2007/synth/nn_results/` (see the README there).*

## Setup

| Run | Instances | Threads per solver run | Time limit |
|---|---|---|---|
| A | 30 small + 30 medium generated, comp01–21 | 3 | 60 s |
| B | 40 small + 40 medium generated | 1 | 120 s |
| C | same as B (`difficulty_small_medium.csv`) | 1 | 240 s |

| Network experiment | Instances | Training |
|---|---|---|
| N1 baseline | 800 generated small instances (MILP labels, 30 s); held-out sets A and B, 200 each | TRM, default config, CPU, 3,000 steps, seeds 0 and 1 |
| N2 clash penalty | same | N1 plus a penalty on curriculum, teacher and same-course clashes (H1/H3), weight 1 (seeds 0, 1) or 5 (seed 0); 1,500 steps |
| N3 controlled variants | held-out A + B with one property changed: half of the curricula removed (400), one room removed (358; 42 became infeasible), an independent copy added (183 instances with ≤ 21 lectures) | none; N1/N2 models evaluated |

- **Solver difficulty** = time to proven optimality (capped at the limit) and gap = (objective − bound) / objective.
- **Network difficulty** = hard violations per lecture of the network's output (each lecture takes its most probable period and room), averaged over all evaluations during training.
- **ρ|size** = Spearman correlation after removing the effect of `n_lectures`.
- **Significance (5 %, two-sided):** |ρ| ≥ 0.36 / 0.31 / 0.22 for n = 30 / 40 / 80; with Bonferroni correction for 16 tests ≥ 0.52 / 0.46 / 0.33. For the network sets (n = 200): ≥ 0.14, after Bonferroni ≥ 0.21.

## Solver difficulty

**1. Size separates the difficulty classes.**
- ρ(`n_lectures`, solve time) is +0.74 to +0.91 across runs and solvers.
- Across all instances, size alone predicts log solve time on held-out instances as well as all features together (run A, CP-SAT: R² 0.85 vs 0.82).

**2. Within a size class, curricula per course is the main additional driver.**

| Group | Solver, run | Measure | ρ\|size `curricula_per_course` | ρ\|size `conflict_density` |
|---|---|---|---|---|
| medium (n = 30) | CP-SAT, A | time | **+0.72** | **+0.59** |
| medium (n = 30) | CP-SAT, A | gap | **+0.68** | **+0.57** |
| medium (n = 40) | MILP, C | gap | **+0.57** | **+0.54** |
| small (n = 40) | MILP, C | time | **+0.49** | +0.23 |

- **Bold** = significant even after Bonferroni correction.
- Within medium instances, adding the features improves held-out prediction: gap R² 0.30 → 0.55 (MILP, C); log-time R² 0.39 → 0.49 (CP-SAT, A).

**3. The other features add little beyond size.**
- `room_fill`, `max_group_load`, `mean_availability` and `capacity_tightness` show no consistent effect once size is controlled for.
- `min_availability` has a weak but consistent effect (ρ|size −0.12 to −0.33): fewer available periods means harder.
- Raw pooled correlations mislead. `conflict_density` has a raw ρ of about −0.7 (looks like "easier") but ρ|size of 0 to +0.4, because small instances are dense by construction.

**4. The solver setup changes which instances look hard.**
- **3 threads, 60 s (A):** CP-SAT proves 23/30 medium instances optimal, the MILP 4/30.
- **1 thread, 240 s (C):** CP-SAT proves 1/40, the MILP 3/40. The MILP finds the better schedule on 36/37 medium instances, while CP-SAT's lower bound stays at 0 (gap 1.0).
- **120 s → 240 s barely matters:** MILP optima on small instances go from 34 to 35/40, medium is unchanged, and the MILP's median medium gap drops from 0.63 to 0.47.
- **Real instances (A):** CP-SAT proves none optimal. The MILP proves comp01 and comp11 optimal and finds no solution on 15/21.
- **Consequence:** proof time is a usable difficulty label for small instances. For medium instances run single-threaded, only the MILP gap is informative.

## Network difficulty

**5. Multiple optimal solutions.**
- **Count:** the 5 stored training instances in `data/ITC2007/synth` (15–32 lectures) each have **at least 10,000 distinct optimal schedules**. The duplicate-free pool hit its cap of 10,000 every time, so the true numbers are higher.
- **No lecture's period is fixed** across the optima. Two random optima differ in 65–79 % of lecture periods, and each course spreads its lectures over 3.7–5.9× as many periods as it needs. Rooms are nearly fixed (1.0–1.7 effective rooms per course).
- **On the 100 held-out instances checked, 84–86 % of each course's available periods occur in at least one of up to 1,000 optima.** For a single lecture almost every period is "right"; only the combination decides.
- **Genuinely different schedules, not symmetric copies:** unavailabilities fix the order of days in 77/80 generated instances, and equal-capacity rooms allow at most about 8 relabelings.
- **Effect on training:** copying one stored optimum leaves part of the loss impossible to remove (held-out label loss is mostly noise: seed agreement 0.06–0.15). Because ambiguity is this high everywhere, it does not make some instances harder than others (finding 7).

**6. The input encoding hides part of the conflict structure.**
- Conflict visibility in the model's group tags (`nn_tag_auc`, 0.5 = chance): median 0.82 for small, 0.75 for medium and 0.75 (0.69–0.81) for real instances.
- It falls as `curricula_per_course` rises (rank correlation −0.94 among medium, −0.86 among real instances).
- For the network's errors it matters little: ρ|size −0.28 (finding 7), and −0.16 once the number of possible clashes is accounted for.

**7. What makes an instance hard for the network, and why.**

*Result (N1).* The network's difficulty is reproducible: seed agreement 0.87 / 0.91 on A / B. It is predicted by how many clashes are possible, not by size, solver difficulty or ambiguity:

| Measure | ρ\|size, A | ρ\|size, B |
|---|---|---|
| Violations of a random timetable per lecture (`nn_random_guess_hard`) | **+0.81** | **+0.79** |
| `curricula_per_course` | **+0.78** | **+0.75** |
| `conflict_density` | **+0.72** | **+0.69** |
| Mean local load (conflicting lectures ÷ available periods) | **+0.71** | **+0.71** |
| Conflict visibility in the group tags | −0.28 | −0.28 |
| MILP gap / MILP time | +0.25 / +0.20 | +0.25 / +0.26 |
| Pool ambiguity (plausible periods, clash ratio) | ≈ 0 | ≈ 0 |
| `n_lectures` (plain ρ) | ≈ 0 | ≈ 0 |

- **Prediction of B from a fit on A** (Spearman of predicted vs actual): size 0.01; MILP time + gap 0.24; random-timetable violations alone 0.75; plus `curricula_per_course` and `conflict_density` 0.85. Seed agreement on B (0.91) is the ceiling for any measure.
- Too noisy to use as difficulty labels: violations at a single evaluation (seed agreement 0.38–0.46), held-out label loss (0.06–0.15), gain from the recursion (≈ 0).
- The network clashes more than chance: 1.0 hard violations per lecture vs 0.5 for a random timetable; 0–3 of 400 outputs are feasible. Most of it is curriculum and teacher clashes (H3: 0.71 of 1.1 per lecture).

*Tests of the explanation (N2, N3):*

| Claim | Test | Result |
|---|---|---|
| More possible clashes cause more errors | Remove half of the curricula; lectures, rooms and periods unchanged | Violations −23 % (1.09 → 0.84 per lecture; lower on 73 % of 400; p < 10⁻²⁷), almost all curriculum/teacher clashes (0.70 → 0.47). −20 % to −25 % for each of the six models, **including the untrained one** |
| Size itself doesn't matter | Add an independent copy: twice the lectures, same clashes per lecture | Violations per lecture 1.12 → 1.02; in no model higher |
| Ambiguity doesn't drive the errors | Same curriculum removal | The errors fall although ambiguity doesn't: more pools reach 1,000 optima (303 → 388 of 400), spread over periods ≈ unchanged (6.0 → 5.7) |
| The missing clash penalty contributes, but is not the main reason | Train with a clash penalty (N2), compare at 1,500 steps | Violations −11 % (weight 1) and −25 % (weight 5), mostly curriculum clashes (0.75 → 0.59). Still 1.7× a random timetable, at most 3/400 feasible, and **the same instances stay hard** (ρ\|size of random-timetable violations +0.77 to +0.82 in every run). The penalty runs are less sure (top probability 0.55 vs 0.72 for the baseline, which trained twice as long) but still pile lectures into as few periods |
| The network piles lectures into few periods | Count periods used | 13.4 distinct periods per instance vs 18.3 in the label; 42 % of lectures sit in its 3 busiest periods (label 25 %) |
| Feasibility comes from decoding; the network's knowledge shows in quality | Decode the same output with conflict-aware greedy; controls: random preferences, untrained network | Feasible: trained 345–348/400, **random preferences 356**, untrained 313. Soft cost on the 285 instances all of them solve: trained 79–80 (better than random on 86 %), random 125, untrained 131, MILP 12 |
| After coordinated decoding, rooms become the bottleneck | Remove one room; correlate failures after greedy with the measures | One room fewer: argmax violations almost unchanged (curriculum clashes 0.70 → 0.70), greedy feasibility 94 % → 65 % for every trained model (untrained 86 % → 54 %); the MILP is not slowed (median 2.2 → 1.4 s). Greedy failures follow `room_fill` (ρ\|size +0.48 on A and B) and `capacity_tightness` (+0.36 to +0.41), not curricula; the soft gap to the optimum follows `curricula_per_course` (+0.45 / +0.49) |
| Solver-hard is not network-hard | Compare per instance | ρ(network difficulty, MILP time) +0.19. 68 of the 100 instances hardest for the network were solved to proven optimality in 60 s (unproven: 32 % of the hardest quarter, 7 % of the easiest). Doubling the size makes the MILP slower on 87 % of instances (median 0.8 → 4.7 s; proven optimal 166 → 138 of 183), while the network's violations per lecture go down (1.11 → 1.02) |

- **Why, in short:** the network decides every lecture separately. Whatever it has learned, these decisions are not coordinated, so the number of clashes grows with the number of conflicting pairs. That holds even for an untrained network. More training and a clash penalty lower the rate but not this dependence. Ambiguity is extreme everywhere: almost every period is optimal for some timetable. So it neither separates instances nor distinguishes a trained network from an untrained one lecture by lecture. What the trained network knows only shows once a decoder coordinates the lectures.
- **Consequences:**
  - **Curriculum, current argmax model:** order instances within a size class by random-timetable violations per lecture (no solver needed) or by `curricula_per_course`, not by size, solver time, ambiguity or evaluation loss.
  - **Curriculum, with a coordinating decoder:** for feasibility order by `room_fill` / `capacity_tightness`; for solution quality by `curricula_per_course`.
  - **Model:** the biggest lever is coordinated decoding (sequential, or iterative repair), not more penalty terms.
- **Limits:** small instances only; one architecture; CPU training of 3,000 (N1) or 1,500 (N2) steps; weight 5 with one seed. Training instances are fitted much better (0.45 vs 1.02 violations per lecture) but too inconsistently between seeds to rank (agreement ≤ 0.18).

## Data issues (fixed)
- **MILP-generated files can now be encoded.** `solve_pool` also returns `assignment` (the best solution found), like `solve`. `ctt_encode.py` falls back to the first stored solution for older pool files.
- **The MILP solution pool no longer contains duplicates.** They differed only in the minimum-working-days helper variables ("day used"), which could be 0 or 1 at equal cost once a course already met its minimum. `solve_pool` now pins them (`exact_day_used=True`) and drops any repeated timetable. gen002 went from 47 distinct schedules among 1000 pool entries to 1000 of 1000. The files in `data/ITC2007/synth` predate the fix and still contain duplicates.

## Caveats
- One run per instance and solver. Parallel single-threaded runs on a CPU with fast and slow cores (Core Ultra 7 155U) make times noisy.
- The solver findings (1–4) are associations: the generator varies several features together. For the network, the variant tests (N3) and the clash penalty (N2) change one thing at a time, so the effects of curricula, rooms, size and the penalty are causal for this model.
- Pool-based measures use the first solutions the MILP finds (up to 1,000), not a uniform sample of all optima; their ranking of instances is stable between 1,000 and 5,000 solutions (rank agreement 0.65–1.0).
