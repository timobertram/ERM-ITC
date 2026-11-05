import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from ortools.sat.python import cp_model

from dataset.ctt_dataset import collate, expand_paths, load_examples
from dataset.ctt_encode import MAX_PERIODS_PER_DAY
from dataset.ctt_parser import parse_ctt
from dataset.ctt_solver import build_model, evaluate_assignment
from models.recursive_reasoning.trm_ctt import TRMCTT, TRMCTTConfig


@torch.no_grad()
def predict(model, batch, sup_steps):
    """Argmax slot per lecture after sup_steps."""
    z = model.init_z(batch)
    for _ in range(sup_steps):
        z, logits = model(z, batch)
    return logits.argmax(-1).cpu().numpy()


def decode(slots, problem, n_rooms):
    """Slot indices -> period/room dicts for evaluate_assignment."""
    periods, rooms = np.divmod(slots, n_rooms)
    lectures = [(cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)]
    room_ids = list(problem.rooms)
    period_of, room_of = {}, {}
    for i, lec in enumerate(lectures):
        day, slot = divmod(int(periods[i]), MAX_PERIODS_PER_DAY)
        period_of[lec] = day * problem.periods_per_day + slot
        room_of[lec] = room_ids[rooms[i]]
    return period_of, room_of


class _Trace(cp_model.CpSolverSolutionCallback):
    """Records time and objective of the first CP-SAT solution."""
    def __init__(self):
        """No solution seen yet."""
        super().__init__()
        self.first = None

    def on_solution_callback(self):
        """Keep the first solution's time and objective."""
        if self.first is None:
            self.first = (self.WallTime(), self.ObjectiveValue())


def cpsat(problem, hint, time_limit, workers):
    """CP-SAT from scratch (hint=None) or warm-started from a model assignment."""
    model, lectures, period_var, room_var, room_ids = build_model(problem)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    if hint is not None:
        period_of, room_of = hint
        for lec in lectures:
            model.AddHint(period_var[lec], period_of[lec])
            model.AddHint(room_var[lec], room_ids.index(room_of[lec]))
        solver.parameters.repair_hint = True
    trace = _Trace()
    status = solver.Solve(model, trace)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return dict(status=solver.StatusName(status), objective=solver.ObjectiveValue() if feasible else None,
                first_time=trace.first[0] if trace.first else None, first_obj=trace.first[1] if trace.first else None)


def run_instance(ctt_path, hint, time_limit, workers):
    """Hinted and unhinted CP-SAT on one instance."""
    problem = parse_ctt(ctt_path)
    return cpsat(problem, hint, time_limit, workers), cpsat(problem, None, time_limit, workers)


def fmt(x, spec=".0f"):
    """Format or '-' for missing."""
    return "-" if x is None else format(x, spec)


def main():
    """Compare model, hinted CP-SAT and plain CP-SAT on an eval set."""
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="config/ctt_synth3_all.yaml")
    cli.add_argument("--checkpoint", default=None, help="default: <training.checkpoint>_best.pt")
    cli.add_argument("--set", default="real", help="name in data.eval")
    cli.add_argument("--limit", type=int, default=None)
    cli.add_argument("--sup-steps", type=int, default=None)
    cli.add_argument("--time-limit", type=float, default=10.0)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--jobs", type=int, default=8)
    args = cli.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpt_path = args.checkpoint or str(Path(cfg["training"]["checkpoint"]).with_stem(Path(cfg["training"]["checkpoint"]).stem + "_best"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = TRMCTTConfig(**ckpt["config"])
    model = TRMCTT(config).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    sup_steps = args.sup_steps or config.sup_steps

    paths = expand_paths(cfg["data"]["eval"][args.set])[:args.limit]
    ctt_paths = [p.removesuffix("_encoded.npz") + ".ctt" for p in paths]
    rng = np.random.default_rng(cfg["training"]["seed"])
    batch = {k: v.to(device) for k, v in collate(load_examples(paths), rng, cfg["data"].get("lecture_tag", "course")).items()}
    t0 = time.perf_counter()
    slots = predict(model, batch, sup_steps)
    infer_ms = (time.perf_counter() - t0) * 1000 / len(paths)

    n_rooms = batch["room_mask"].shape[1]
    hints, preds = [], []
    for row, p in enumerate(ctt_paths):
        problem = parse_ctt(p)
        hint = decode(slots[row], problem, n_rooms)
        hints.append(hint)
        preds.append(evaluate_assignment(problem, *hint))

    print(f"{ckpt_path} (step {ckpt['step']}), {args.set}: {len(paths)} instances, sup_steps={sup_steps}, "
          f"inference {infer_ms:.1f} ms/instance, CP-SAT {args.time_limit:.0f}s x {args.workers} workers")
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run_instance, ctt_paths, hints, [args.time_limit] * len(paths), [args.workers] * len(paths)))

    print(f"\n{'instance':10s} {'model hard':>10s} {'model soft':>10s} | {'hint t1st':>9s} {'hint 1st':>8s} {'hint end':>8s} | "
          f"{'cold t1st':>9s} {'cold 1st':>8s} {'cold end':>8s}")
    for p, pred, (hint, cold) in zip(ctt_paths, preds, results):
        print(f"{Path(p).stem:10s} {pred['hard_violations']:10d} {pred['soft_objective']:10d} | "
              f"{fmt(hint['first_time'], '8.2f')}s {fmt(hint['first_obj']):>8s} {fmt(hint['objective']):>8s} | "
              f"{fmt(cold['first_time'], '8.2f')}s {fmt(cold['first_obj']):>8s} {fmt(cold['objective']):>8s}")

    def mean(xs):
        """Mean over non-missing values."""
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    print(f"\n{'mean':10s} {fmt(mean([p['hard_violations'] for p in preds]), '10.1f')} {fmt(mean([p['soft_objective'] for p in preds]), '10.1f')} | "
          f"{fmt(mean([h['first_time'] for h, _ in results]), '8.2f')}s {fmt(mean([h['first_obj'] for h, _ in results]), '8.1f')} "
          f"{fmt(mean([h['objective'] for h, _ in results]), '8.1f')} | "
          f"{fmt(mean([c['first_time'] for _, c in results]), '8.2f')}s {fmt(mean([c['first_obj'] for _, c in results]), '8.1f')} "
          f"{fmt(mean([c['objective'] for _, c in results]), '8.1f')}")
    for name, i in [("hint", 0), ("cold", 1)]:
        print(f"{name} feasible: {sum(r[i]['objective'] is not None for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
