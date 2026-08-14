# erm-itc

TRM (Tiny Recursive Model) applied to ITC2007 Track 3 (Curriculum-Based
Course Timetabling, "CB-CTT") instances.

Format reference: the official ITC2007 spec, mirrored at
[Docheinstein/itc2007-cct](https://github.com/Docheinstein/itc2007-cct).

## Workflow

Train on the real instances (`config/ctt.yaml`: trains on `comp01`-`comp20`,
evaluates on `comp21`):
```bash
python -m dataset.ctt_encode 'comp*' --data-dir data/ITC2007/real
python pretrain_ctt.py --config config/ctt.yaml
```

Train on generated synthetic instances instead (`config/ctt_synth.yaml`:
evaluates on all 21 real instances) -- generate, encode both sides, train:
```bash
python -m dataset.ctt_generator --count 1000 --size small --jobs 8
python -m dataset.ctt_encode 'gen*' --data-dir data/ITC2007/synth
python -m dataset.ctt_encode 'comp*' --data-dir data/ITC2007/real   # the eval set
python pretrain_ctt.py --config config/ctt_synth.yaml
```

Every `pretrain_ctt.py` run needs its `data.paths`/`eval_sets` `.npz` files
to already exist -- `dataset.ctt_encode` only writes `<name>_encoded.npz`
when explicitly run against a directory that has matching `.ctt` +
`_solution.json` pairs, it isn't run automatically by anything else.

## Pipeline

1. **Parse** a `.ctt` instance -- `dataset/ctt_parser.py::parse_ctt`

2. **Solve** an instance to generate ground-truth labels (CP-SAT via
   `ortools`):
   ```bash
   python -m dataset.ctt_solver data/ITC2007/real/comp01.ctt --time-limit 60 --workers 8 --out data/ITC2007/real/comp01_solution.json
   ```

3. **Encode** one or more solved instances into a shared period vocab +
   dense feature tensors:
   ```bash
   python -m dataset.ctt_encode comp01 comp02 --data-dir data/ITC2007/real
   # or, with a glob:
   python -m dataset.ctt_encode 'comp*' --data-dir data/ITC2007/real
   ```
   Writes `<name>_encoded.npz` next to each instance.

4. **Train** (persistent ACT slots over random mini-batches; config-driven,
   see `config/ctt.yaml` for the full set of knobs -- data paths, wandb
   project/run name, held-out eval sets, checkpointing, model
   hyperparameters, and which model class to train via
   `model.module`/`class_name`/`loss_head_class_name`, e.g. `trm_ctt`/
   `TRMCTT`/`TRMCTTLossHead` for `models/recursive_reasoning/trm_ctt.py`):
   ```bash
   python pretrain_ctt.py --config config/ctt.yaml
   ```
   The base config trains on the 20 real instances `comp01`-`comp20` and
   holds out `comp21` for eval. A positional `npz_paths` argument overrides
   `data.paths` from the config if given. Copy `config/ctt.yaml` per
   experiment rather than editing it in place. Logs to Weights & Biases
   (`wandb login` first, or set
   `WANDB_MODE=offline`).

## Model

`models/recursive_reasoning/trm_ctt.py::TRMCTT` -- one recursive latent per
*lecture*, refined by ordinary dense self-attention. Same-curriculum/
same-teacher relations are baked into each lecture's feature vector at
encode time (see `dataset/ctt_encode.py`). Two output heads read from the
same recursive latent `z`, both via query/key dot product: periods score
against a learned global embedding table (shared vocab across instances),
rooms score against that instance's own room table (rooms don't generalize
across instances, so there's no fixed head for them). Since lectures within
a course are interchangeable, training uses a Hungarian-matched period loss
(`models/recursive_reasoning/trm_ctt.py::_match_period_targets`) plus a
soft room-collision penalty.

Besides accuracy against the solver's specific solution, `pretrain_ctt.py`
also logs `dataset/ctt_solver.py::evaluate_assignment` metrics -- the exact
H1-H4/S1-S4 constraint counts for the model's own proposed assignment,
independent of any reference solution (a model can find a *different* valid
solution than the solver's). Verified to reproduce the solver's own
objective value exactly when fed the solver's solution.

## Constraints (H1-H4, S1-S4)

The official ITC2007 Track 3 spec, enforced identically by `ctt_solver.py`
(as CP-SAT constraints) and `ctt_solver.py::evaluate_assignment` (as a
plain check against any assignment, e.g. a model's predictions):

Hard (must never be violated in a feasible solution):
- **H1** -- a course's own lectures must all land on distinct periods.
- **H2** -- no two lectures may share the same (period, room).
- **H3a** -- lectures of courses in the same curriculum must all land on
  distinct periods.
- **H3b** -- lectures of courses taught by the same teacher must all land
  on distinct periods.
- **H4** -- a lecture can't be scheduled at a period its course is marked
  unavailable for.

Soft (allowed, but penalized in the objective -- weights in `ctt_solver.ALPHA`):
- **S1** (weight 1) -- RoomCapacity: 1 point per student over the chosen
  room's capacity.
- **S2** (weight 5) -- MinimumWorkingDays: 5 points per day short of a
  course's minimum spread across distinct days.
- **S3** (weight 2) -- CurriculumCompactness: 2 points per lecture with no
  same-curriculum neighbor in the adjacent slot(s) of the same day.
- **S4** (weight 1) -- RoomStability: 1 point per distinct room used beyond
  the first, per course.

`evaluate_assignment`'s hard-violation *counts* (H1-H4) are "excess"
assignments per group, not a broken/not-broken flag -- e.g. 3 lectures of
one course sharing a single period count as 2 violations, not 1.
`soft_objective` is the same `ALPHA`-weighted sum `ctt_solver.py` minimizes,
so it's directly comparable to the `objective` value recorded in each
`*_solution.json`.

## Data

`data/ITC2007/real/` ships the 21 official ITC2007 competition instances
(`comp01.ctt`-`comp21.ctt`) plus their solved `_solution.json` files
(produced by `dataset/ctt_solver.py`) -- tracked in git despite the general
`data/` gitignore rule (see `.gitignore`). Encoded `.npz` outputs are not
tracked; regenerate them with `dataset/ctt_encode.py`.

`data/ITC2007/synth/` holds synthetic instances from `dataset/ctt_generator.py`
(same format, feasibility-checked, solved via `ctt_solver.py`) for a bigger
training corpus -- entirely untracked (matches the general `data/`
gitignore rule):

```bash
python -m dataset.ctt_generator --count 50 --size small --out-dir data/ITC2007/synth
```

`--size` is `small`, `medium`, or `large` (`large` covers real data's actual
room-capacity/class-size range, up to 450/440 -- `small`/`medium` top out at
150). Add `--jobs N` to solve instances in parallel.

## Setup

```bash
uv sync   # or: pip install -e .
```

Dependencies: `torch`, `numpy`, `pydantic`, `ortools` (CP-SAT solver),
`scipy` (Hungarian matching), `wandb` (experiment logging), `pyyaml`
(config files), `tqdm` (generator progress bar).
