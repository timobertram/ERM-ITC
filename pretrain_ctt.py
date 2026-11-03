"""Config-driven training loop for models in models/recursive_reasoning/.

See config/ctt.yaml for the full set of knobs (data paths, wandb project/run
name, held-out eval sets, checkpointing, model hyperparameters, and which
model class to train -- model.module/class_name/loss_head_class_name).

A fixed number of persistent ACT "slots" (training.batch_size), each with
its own carry. Every step, a fresh random candidate instance is offered to
each slot, but a slot only adopts it once its own `halted` flag fires.

Also logs H1-H4/S1-S4 constraint violations for the model's own proposed
assignment (dataset.ctt_solver), gated to log_every/eval_every since it's
plain Python and re-parses the .ctt files. training.instance_log (a CSV
path, overwritten at the start of each run) additionally records every eval
instance's own results at every eval, for per-instance difficulty analysis
(see dataset/ctt_difficulty.py).
"""
import argparse
import csv
import importlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from dataset.ctt_dataset import expand_paths, load_batch
from dataset.ctt_solver import instance_solution_metrics, solution_metrics


def _log_metrics(prefix, line_prefix, loss, metrics, step, extra=""):
    print(f"{line_prefix}loss={loss.item():.4f}  "
          f"period_set_acc={metrics['period_set_accuracy'].item():.4f}  "
          f"room_acc={metrics['room_accuracy'].item():.4f}  "
          f"exact_acc={metrics['exact_accuracy'].item():.4f}{extra}")
    wandb.log({
        f"{prefix}/loss": loss.item(),
        f"{prefix}/period_set_accuracy": metrics["period_set_accuracy"].item(),
        f"{prefix}/room_accuracy": metrics["room_accuracy"].item(),
        f"{prefix}/exact_accuracy": metrics["exact_accuracy"].item(),
    }, step=step)


def _log_solution_metrics(prefix, sol_metrics, step):
    print(f"  [{prefix} solution] hard_violations={sol_metrics['hard_violations']:.2f} "
          f"(h1={sol_metrics['h1']:.2f} h2={sol_metrics['h2']:.2f} h3a={sol_metrics['h3a']:.2f} "
          f"h3b={sol_metrics['h3b']:.2f} h4={sol_metrics['h4']:.2f})  "
          f"soft_objective={sol_metrics['soft_objective']:.2f} "
          f"(s1={sol_metrics['s1']:.2f} s2={sol_metrics['s2']:.2f} s3={sol_metrics['s3']:.2f} s4={sol_metrics['s4']:.2f})")
    wandb.log({f"{prefix}/{k}": v for k, v in sol_metrics.items()}, step=step)


def _masked_mean(x, mask):
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _instance_rows(set_name, step, batch, outputs, first_outputs, ctt_paths, problem_cache):
    """One row per instance of an eval set (argmax decoding, as in
    solution_metrics):

      hard_violations, soft_objective, h1-h4   after the last ACT step.
      hard_step1           hard violations after the first ACT step --
                           compare with hard_violations to see whether
                           recursion helps on this instance.
      period_target_prob,  mean probability of the (Hungarian-matched)
      room_target_prob     target per lecture -- data cartography's
                           "confidence" when tracked over evals.
      period_loss          mean -log period_target_prob.
      period_el2n          mean L2 distance between the period distribution
                           and the one-hot matched target (EL2N, Paul et
                           al. 2021).
      plausible_periods    mean exp(entropy) of the predicted period
                           distribution: how many periods the model spreads
                           over (compare ctt_difficulty's nn_plausible_periods).
    """
    mask = batch["node_mask"].float()
    period_prob, room_prob = outputs["period_prob"], outputs["room_prob"]
    matched = outputs["matched_period_label"]
    target_period_prob = period_prob.gather(-1, matched.unsqueeze(-1)).squeeze(-1)
    target_room_prob = room_prob.gather(-1, batch["room_label"].unsqueeze(-1)).squeeze(-1)
    el2n = (period_prob - F.one_hot(matched, period_prob.shape[-1]).float()).norm(dim=-1)
    entropy = -(period_prob * period_prob.clamp_min(1e-12).log()).sum(dim=-1)
    per_lecture = dict(
        period_target_prob=target_period_prob,
        room_target_prob=target_room_prob,
        period_loss=-target_period_prob.clamp_min(1e-8).log(),
        period_el2n=el2n,
        plausible_periods=entropy.exp(),
    )
    per_instance = {k: _masked_mean(v, mask).cpu().numpy() for k, v in per_lecture.items()}

    final = instance_solution_metrics(period_prob, room_prob, ctt_paths, problem_cache)
    first = instance_solution_metrics(first_outputs["period_prob"], first_outputs["room_prob"], ctt_paths, problem_cache)
    n_lectures = batch["node_mask"].sum(dim=1).cpu().numpy()
    rows = []
    for row, ctt_path in enumerate(ctt_paths):
        rows.append(dict(
            set=set_name, step=step, instance=os.path.splitext(os.path.basename(ctt_path))[0],
            n_lectures=int(n_lectures[row]),
            **{k: final[row][k] for k in ("hard_violations", "h1", "h2", "h3a", "h3b", "h4", "soft_objective")},
            hard_step1=first[row]["hard_violations"],
            **{k: float(v[row]) for k, v in per_instance.items()},
        ))
    return rows


def _append_csv(path, rows):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def _run_eval(model, eval_batches, eval_ctt_paths, problem_cache, step, instance_log=None):
    model.eval()
    with torch.no_grad():
        for name, eval_batch in eval_batches.items():
            carry = model.initial_carry(eval_batch)
            all_halted = False
            first_outputs = None
            while not all_halted:
                carry, loss, metrics, outputs, all_halted = model(carry, eval_batch)
                first_outputs = first_outputs or outputs
            _log_metrics(name, f"  [{name} @ step {step}] ", loss, metrics, step)
            sol_metrics = solution_metrics(
                outputs["period_prob"], outputs["room_prob"], eval_ctt_paths[name], problem_cache
            )
            _log_solution_metrics(name, sol_metrics, step)
            if instance_log:
                _append_csv(instance_log, _instance_rows(
                    name, step, eval_batch, outputs, first_outputs, eval_ctt_paths[name], problem_cache
                ))
    model.train()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("npz_paths", nargs="*", default=None, help="overrides the config file's data.paths if given")
    cli.add_argument("--config", default="config/ctt.yaml")
    args = cli.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]

    model_module = importlib.import_module(f"models.recursive_reasoning.{model_cfg['module']}")
    ModelClass = getattr(model_module, model_cfg["class_name"])
    LossHeadClass = getattr(model_module, model_cfg["loss_head_class_name"])

    wandb.init(
        project=train_cfg.get("project_name") or "ctt",
        name=train_cfg.get("run_name"),
        config=cfg,
        settings=wandb.Settings(_disable_stats=True),
    )

    paths = args.npz_paths or data_cfg["paths"]
    device = torch.device(train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    full_batch = {k: v.to(device) for k, v in load_batch(paths).items()}
    total_instances = full_batch["lecture_course_features"].shape[0]
    batch_size = train_cfg.get("batch_size") or total_instances

    # maps each full_batch row to its source .ctt file, for solution_metrics
    instance_ctt_paths = [p[: -len("_encoded.npz")] + ".ctt" for p in expand_paths(paths)]
    problem_cache: dict = {}

    model_config = dict(
        hidden_size=model_cfg["hidden_size"], expansion=model_cfg["expansion"],
        num_heads=model_cfg["num_heads"], head_dim=model_cfg["head_dim"],
        course_feature_dim=full_batch["lecture_course_features"].shape[-1],
        random_id_dim=full_batch["lecture_random_id"].shape[-1],
        group_tag_dim=full_batch["lecture_group_tag"].shape[-1],
        room_random_id_dim=full_batch["room_random_id"].shape[-1],
        period_vocab_size=full_batch["period_candidate_mask"].shape[-1],
        L_layers=model_cfg["l_layers"], H_cycles=model_cfg["h_cycles"], L_cycles=model_cfg["l_cycles"],
        halt_max_steps=model_cfg["halt_max_steps"], halt_exploration_prob=model_cfg["halt_exploration_prob"],
        halt_max_prob=model_cfg["halt_max_prob"], forward_dtype=model_cfg["forward_dtype"],
    )

    torch.manual_seed(train_cfg.get("seed", 0))  # weight init and ACT exploration; data order uses rng below
    with torch.device(device):
        model = LossHeadClass(ModelClass(model_config), collision_weight=train_cfg.get("collision_weight", 1.0))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.0)

    # named held-out eval sets, all checked at the same cadence; eval_paths
    # (singular, unnamed) is a shortcut for one set named "held-out"
    eval_every = train_cfg.get("eval_every")
    eval_sets = {}
    if train_cfg.get("eval_paths"):
        eval_sets["held-out"] = train_cfg["eval_paths"]
    eval_sets.update(train_cfg.get("eval_sets") or {})
    eval_batches = {name: {k: v.to(device) for k, v in load_batch(p).items()} for name, p in eval_sets.items()}
    eval_ctt_paths = {
        name: [p[: -len("_encoded.npz")] + ".ctt" for p in expand_paths(p_list)]
        for name, p_list in eval_sets.items()
    }
    instance_log = train_cfg.get("instance_log")
    if instance_log:
        Path(instance_log).parent.mkdir(parents=True, exist_ok=True)
        Path(instance_log).unlink(missing_ok=True)

    rng = np.random.default_rng(train_cfg.get("seed", 0))
    perm = rng.permutation(total_instances)
    ptr = 0
    candidate_idx = None

    def _next_candidate_batch():
        nonlocal perm, ptr, candidate_idx
        if ptr + batch_size > len(perm):
            perm = rng.permutation(total_instances)
            ptr = 0
        idx = perm[ptr:ptr + batch_size]
        ptr += batch_size
        candidate_idx = idx
        idx_t = torch.tensor(idx.tolist(), device=device, dtype=torch.long)
        return {k: v[idx_t] for k, v in full_batch.items()}

    carry = model.initial_carry(_next_candidate_batch())
    current_row_idx = np.zeros(batch_size, dtype=np.int64)  # which full_batch row each slot currently holds
    print(f"training on {total_instances} instances, {batch_size} persistent ACT slots per step")

    checkpoint_path = train_cfg.get("checkpoint_path")
    checkpoint_every = train_cfg.get("checkpoint_every")

    def _save_checkpoint(step):
        if not checkpoint_path:
            return
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_config": model_config, "state_dict": model.state_dict(), "step": step}, checkpoint_path)
        print(f"  saved checkpoint to {checkpoint_path} (step {step})")

    model.train()
    steps, log_every = train_cfg["steps"], train_cfg["log_every"]
    for step in range(steps):
        halted_before = carry.halted.cpu().numpy()
        batch = _next_candidate_batch()
        current_row_idx = np.where(halted_before, candidate_idx, current_row_idx)

        optimizer.zero_grad()
        carry, loss, metrics, outputs, all_halted = model(carry, batch)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            extra = f"  collision_loss={metrics['collision_loss'].item():.4f}  halted={bool(all_halted.item())}"
            _log_metrics("train", f"step {step:4d}  ", loss, metrics, step, extra=extra)
            wandb.log({"train/collision_loss": metrics["collision_loss"].item()}, step=step)
            sol_metrics = solution_metrics(
                outputs["period_prob"], outputs["room_prob"],
                [instance_ctt_paths[i] for i in current_row_idx], problem_cache,
            )
            _log_solution_metrics("train", sol_metrics, step)

        if eval_batches and eval_every and (step % eval_every == 0 or step == steps - 1):
            _run_eval(model, eval_batches, eval_ctt_paths, problem_cache, step, instance_log)

        if checkpoint_every and (step % checkpoint_every == 0 or step == steps - 1) and step > 0:
            _save_checkpoint(step)

    _save_checkpoint(steps - 1)
    wandb.finish()


if __name__ == "__main__":
    main()
