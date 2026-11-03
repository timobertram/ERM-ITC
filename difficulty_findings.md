# Instance difficulty for CB-CTT: current findings

*2026-11-02. Tool: `dataset/ctt_difficulty.py`: 8 structural features, CP-SAT/MILP runs and rank-correlation analysis. `--nn` adds 11 network-specific measures; `--analyze CSV --instance-log LOG...` relates the measures to per-instance network results logged by `pretrain_ctt.py` (`training.instance_log`).*

## Setup

| Run | Instances | Threads per solver run | Time limit |
|---|---|---|---|
| A | 30 small + 30 medium generated, comp01–21 | 3 | 60 s |
| B | 40 small + 40 medium generated | 1 | 120 s |
| C | same as B (`difficulty_small_medium.csv`) | 1 | 240 s |

- **Difficulty** = time to proven optimality (capped at the limit) and gap = (objective − bound) / objective.
- **ρ|size** = Spearman correlation after removing the effect of `n_lectures`.
- **Significance (5 %, two-sided):** |ρ| ≥ 0.36 / 0.31 / 0.22 for n = 30 / 40 / 80; with Bonferroni correction for 16 tests ≥ 0.52 / 0.46 / 0.33.

## Main findings

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

**5. Multiple optimal solutions (relevant for training the network).**
- **Count:** the 5 stored training instances in `data/ITC2007/synth` (15–32 lectures) each have **at least 10,000 distinct optimal schedules**. The duplicate-free pool hit its cap of 10,000 every time, so the true numbers are higher.
- **No lecture's period is fixed** across the optima. Two random optima differ in 65–79 % of lecture periods, and each course spreads its lectures over 3.7–5.9× as many periods as it needs. Rooms are nearly fixed (1.0–1.7 effective rooms per course).
- **Genuinely different schedules, not symmetric copies:** unavailabilities fix the order of days in 77/80 generated instances, and equal-capacity rooms allow at most about 8 relabelings.
- **Effect on training:** copying one stored optimum leaves part of the loss impossible to remove, and pushes the model to mix optima into schedules that break constraints. Instances that are easy for solvers may be the most ambiguous ones to learn by copying.

**6. The input encoding hides the structure that makes instances hard.**
- Conflict visibility in the model's group tags (AUC, 0.5 = chance): median 0.81 for small, 0.73 for medium, 0.64–0.86 for real instances.
- Among medium instances it falls as `curricula_per_course` rises (rank correlation −0.86). The main driver of solver difficulty also reduces what the network can see.

**7. What makes an instance hard for the network (training experiment).**
- **Setup:**
  - Data: 800 generated small training instances, labelled by the MILP (30 s each).
  - Model: TRM with the default config, trained for 3,000 steps on CPU with two seeds.
  - Held-out sets: A (200 instances, used to find effects) and B (200, used to check that they replicate).
  - Measures from `ctt_difficulty.py --nn --solver milp`; results and checkpoints are in `data/ITC2007/synth/nn_results/`.
- **Network difficulty** = hard violations per lecture of the argmax output, averaged over all 12 evaluations. Its agreement between seeds is 0.87–0.91. Too noisy to use as a difficulty label:
  - violations at the last evaluation alone (agreement 0.38–0.46);
  - held-out period loss (0.06–0.15);
  - gain from recursion (≈ 0).

| Measure | ρ\|size, A | ρ\|size, B |
|---|---|---|
| Violations of a random timetable per lecture (`nn_random_guess_hard`) | **+0.81** | **+0.79** |
| `curricula_per_course` | **+0.78** | **+0.75** |
| `conflict_density` | **+0.72** | **+0.69** |
| Mean local load (conflicting lectures ÷ available periods) | **+0.71** | **+0.71** |
| Conflict visibility in the group tags (`nn_tag_auc`) | −0.28 | −0.28 |
| MILP gap / MILP time | +0.25 / +0.20 | +0.25 / +0.26 |
| Pool ambiguity (plausible periods, clash ratio) | ≈ 0 | ≈ 0 |
| `n_lectures` (plain ρ) | ≈ 0 | ≈ 0 |

- **Prediction on B from a fit on A** (Spearman of predicted vs actual): size 0.01; MILP time + gap 0.24; random-timetable violations alone 0.75; plus `curricula_per_course` and `conflict_density` 0.85. Seed agreement on B is 0.91, the ceiling for any measure.
- **Why:**
  - **The network clashes more than chance.** Decoded by argmax, it makes 1.0 hard violations per lecture on held-out instances, against 0.5 for a random timetable; 0–1 of 400 schedules are feasible.
  - **Most of its violations are curriculum and teacher clashes** (H3: 0.71 of about 1.1 per lecture). Only the imitation loss addresses these; the collision loss covers rooms (H2) only. Its error per instance therefore follows how many clashes are possible.
  - **Ambiguity is high everywhere, so it doesn't separate instances.** The network does not learn the spread over optima either: it puts its probability on about 2.5 periods per lecture, regardless of the instance's 6 plausible ones (ρ +0.02). The ranking of instances by pool measures is stable: 1,000 vs 5,000 optima agree 0.65–1.0.
  - **Solvers and the network struggle with different things.** The MILP struggles to prove optimality and to cut soft costs; the network struggles to avoid hard clashes.
  - **The knowledge is there; independent decoding loses it.** Decoding the same outputs greedily with conflict avoidance gives 345–348 of 400 feasible schedules (first-fit greedy without the network: 264). On instances where both are feasible, the soft cost is 77–79 vs 105 (MILP 12). Greedy still fails mainly where rooms are full (`room_fill` median 0.52–0.54 vs 0.27), which becomes the next difficulty axis once decoding is fixed.
- **Consequence for a curriculum:** within a size class, order instances by random-timetable violations per lecture (no solver needed) or by `curricula_per_course`. Don't order by size, solver time or evaluation loss. This ranking describes the current argmax-decoded model and should be re-measured after changing the loss or the decoding.
- **Limits:** small instances only; one architecture; 3,000 CPU steps. Training instances are fitted much better (0.45 vs 1.02 violations per lecture) but too inconsistently between seeds to rank (agreement ≤ 0.18).

## Data issues (fixed)
- **MILP-generated files can now be encoded.** `solve_pool` also returns `assignment` (the best solution found), like `solve`. `ctt_encode.py` falls back to the first stored solution for older pool files.
- **The MILP solution pool no longer contains duplicates.** They differed only in the minimum-working-days helper variables ("day used"), which could be 0 or 1 at equal cost once a course already met its minimum. `solve_pool` now pins them (`exact_day_used=True`) and drops any repeated timetable. gen002 went from 47 distinct schedules among 1000 pool entries to 1000 of 1000. The files in `data/ITC2007/synth` predate the fix and still contain duplicates.

## Caveats
- One run per instance and solver. Parallel single-threaded runs on a CPU with fast and slow cores (Core Ultra 7 155U) make times noisy.
- The generator varies several features together, so these are associations, not shown causes.
