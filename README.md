# erm-itc

TRM (Tiny Recursive Model) on ITC2007 Track 3 curriculum-based course timetabling.

## Layout

```
dataset/ctt_parser.py     .ctt -> CTTProblem
dataset/ctt_solver.py     CP-SAT solver, evaluate_assignment (H1-H4, S1-S4)
dataset/ctt_generator.py  synthetic instances
dataset/ctt_encode.py     CTTProblem -> .npz
dataset/ctt_dataset.py    .npz -> padded batch
models/common.py          init
models/layers.py          attention, SwiGLU, rms_norm
models/recursive_reasoning/trm_ctt.py   model + constraint loss
pretrain_ctt.py
repair_ctt.py             CP-SAT repair from model predictions
scripts/gen_synth.sh      generate + encode small/medium/large
```

## Model

Lecture tokens and (period, room) slot tokens, refined with H/L recursion
cycles and `sup_steps` deep-supervision steps per batch. Lectures self-attend
with a learned course/curriculum/teacher bias and cross-attend to slots; slots
cross-attend to lectures. Each lecture points to one slot.

With `model.feedback`, the previous step's argmax assignment is fed back into
the input: the chosen slot's embedding and clash counts per lecture, occupancy
per slot.

Lecture inputs: course features plus learned embeddings of the course's index
in the instance and the lecture's index within its course (symmetry breaking),
and an optional random tag (`data.lecture_tag`).

## Objective

H1-H3 and S1-S4 violations per lecture of a straight-through argmax
one-hot slot assignment. No reference solution is used. H4 is enforced by
masking unavailable periods.

`training.objectives` toggles each term (`enabled`) and sets its `weight`, the
step it turns on (`start`) and a linear warmup (`ramp`). Eval uses full weights.

## Usage

```bash
uv sync
python -m dataset.ctt_generator --count 1000 --size small --jobs 8
python -m dataset.ctt_encode data/ITC2007/real/*.ctt data/ITC2007/synth/*.ctt
python pretrain_ctt.py --config config/ctt_synth.yaml
```

`config/ctt.yaml` trains on comp01-comp20 and evaluates on comp21.
