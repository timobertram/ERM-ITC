import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from dataset.ctt_dataset import collate, expand_paths, load_examples
from dataset.ctt_encode import MAX_PERIODS_PER_DAY
from dataset.ctt_parser import parse_ctt
from dataset.ctt_solver import evaluate_assignment
from models.recursive_reasoning.trm_ctt import TRMCTT, TRMCTTConfig, constraint_loss

_problems = {}


def solution_metrics(logits, n_rooms, ctt_paths):
    """Evaluator metrics of the argmax assignment, averaged over instances."""
    periods, rooms = np.divmod(logits.argmax(-1).cpu().numpy(), n_rooms)
    totals = {}
    for row, path in enumerate(ctt_paths):
        if path not in _problems:
            _problems[path] = parse_ctt(path)
        problem = _problems[path]
        room_ids = list(problem.rooms)
        lectures = [(cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)]
        period_of, room_of = {}, {}
        for i, lec in enumerate(lectures):
            day, slot = divmod(int(periods[row, i]), MAX_PERIODS_PER_DAY)
            period_of[lec] = day * problem.periods_per_day + slot
            room_of[lec] = room_ids[rooms[row, i]]
        sol = evaluate_assignment(problem, period_of, room_of)
        sol["feasible"] = int(sol["hard_violations"] == 0)
        for k, v in sol.items():
            totals[k] = totals.get(k, 0) + v / len(ctt_paths)
    return totals


def log(prefix, step, loss, terms, sol, **extra):
    """Print and log to wandb."""
    print(f"[{prefix} {step:5d}] loss={loss:.4f}  hard={sol['hard_violations']:.2f}  soft={sol['soft_objective']:.1f}")
    wandb.log({f"{prefix}/loss": loss,
               **{f"{prefix}/loss_{k}": v.item() for k, v in terms.items()},
               **{f"{prefix}/{k}": v for k, v in sol.items()},
               **{f"{prefix}/{k}": v for k, v in extra.items()}}, step=step)


def objective_weights(objectives, step=None):
    """Loss weights at step after start/ramp; full weights if step is None."""
    weights = {}
    for k, o in objectives.items():
        t = 0 if step is None else step - o.get("start", 0)
        if not o.get("enabled", True) or t < 0:
            continue
        ramp = 0 if step is None else o.get("ramp", 0)
        weights[k] = o["weight"] * (min(1.0, (t + 1) / ramp) if ramp else 1.0)
    return weights


def main():
    """Train with deep supervision and periodic eval/checkpointing."""
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="config/ctt.yaml")
    cfg = yaml.safe_load(open(cli.parse_args().config))
    tc = cfg["training"]

    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"])
    device = torch.device(tc["device"])
    wandb.init(project=tc["project"], name=tc.get("run_name"), config=cfg)

    def to_ctt(paths):
        """Encoded .npz paths -> .ctt paths."""
        return [p.removesuffix("_encoded.npz") + ".ctt" for p in paths]

    def to_device(batch):
        """Move a batch to the device."""
        return {k: v.to(device) for k, v in batch.items()}

    paths = expand_paths(cfg["data"]["paths"])
    examples, ctt_paths = load_examples(paths), to_ctt(paths)
    lecture_tag = cfg["data"].get("lecture_tag", "course")
    eval_rng = np.random.default_rng([tc["seed"], 1])
    evals = {}
    for name, patterns in cfg["data"].get("eval", {}).items():
        eval_paths = expand_paths(patterns)
        evals[name] = to_device(collate(load_examples(eval_paths), eval_rng, lecture_tag)), to_ctt(eval_paths)

    config = TRMCTTConfig(
        lecture_dim=examples[0]["lecture_features"].shape[-1],
        room_dim=examples[0]["room_features"].shape[-1],
        period_vocab=examples[0]["period_mask"].shape[-1],
        **cfg["model"],
    )
    model = TRMCTT(config).to(device)
    if tc.get("init_from"):
        model.load_state_dict(torch.load(tc["init_from"], map_location=device)["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=0.0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["steps"], eta_min=tc["lr_min"])
             if tc.get("lr_min") is not None else None)
    objectives = tc["objectives"]
    for k, o in objectives.items():
        if unknown := set(o) - {"enabled", "weight", "start", "ramp"}:
            raise ValueError(f"unknown keys for objective {k}: {unknown}")
    if not objective_weights(objectives, 0):
        raise ValueError("no objective active at step 0")
    loss_fn = lambda logits, b, w: constraint_loss(model.assign(logits), b, w)

    def save(step, suffix=""):
        """Save checkpoint, optionally with a suffix (_best, _feasible)."""
        if tc.get("checkpoint"):
            path = Path(tc["checkpoint"])
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": asdict(config), "state_dict": model.state_dict(), "step": step},
                       path.with_stem(path.stem + suffix))

    @torch.no_grad()
    def evaluate(step):
        """Run all eval sets and return their metrics."""
        model.eval()
        metrics = {}
        for name, (batch, paths) in evals.items():
            z = model.init_z(batch)
            for _ in range(config.sup_steps):
                z, p = model(z, batch)
            loss, terms = loss_fn(p, batch, objective_weights(objectives))
            sol = solution_metrics(p, batch["room_mask"].shape[1], paths)
            log(name, step, loss.item(), terms, sol)
            metrics.update({f"{name}/{k}": v for k, v in sol.items()})
        model.train()
        return metrics

    best = float("inf")
    best_feasible = 0

    n = len(examples)
    bs = tc.get("batch_size") or n
    order, ptr = rng.permutation(n), 0
    for step in range(tc["steps"]):
        if step % config.sup_steps == 0:
            if ptr + bs > n:
                order, ptr = rng.permutation(n), 0
            idx = order[ptr:ptr + bs]
            ptr += bs
            batch = to_device(collate([examples[i] for i in idx], rng, lecture_tag))
            z = model.init_z(batch)

        z, p = model(z, batch)
        weights = objective_weights(objectives, step)
        loss, terms = loss_fn(p, batch, weights)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip") or float("inf")).item()
        opt.step()
        if sched:
            sched.step()

        last = step == tc["steps"] - 1
        if step % tc["log_every"] == 0 or last:
            log("train", step, loss.item(), terms, solution_metrics(p, batch["room_mask"].shape[1], [ctt_paths[i] for i in idx]),
                grad_norm=grad_norm, lr=opt.param_groups[0]["lr"], **{f"weight_{k}": v for k, v in weights.items()})
        if evals and (step % tc["eval_every"] == 0 or last):
            metrics = evaluate(step)
            if tc.get("best_metric") and (m := sum(metrics[k] for k in tc["best_metric"])) < best:
                best = m
                save(step, "_best")
                wandb.run.summary.update({"best/step": step, "best/metric": m})
            if (k := tc.get("best_feasible")) and metrics[k] > best_feasible:
                best_feasible = metrics[k]
                save(step, "_feasible")
                wandb.run.summary.update({"best_feasible/step": step, "best_feasible/value": metrics[k]})
        if tc.get("checkpoint_every") and step and step % tc["checkpoint_every"] == 0:
            save(step)

    save(tc["steps"] - 1)
    wandb.finish()


if __name__ == "__main__":
    main()
