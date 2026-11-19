#!/usr/bin/env python3
"""Train a total-MBE(3) energy-conserving HydraGNN QM/MM model."""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from mbe3_dataset import MBE3Dataset, read_split_files
from mbe3_model import MBE3QMMMModel, MODEL_ARCHITECTURE_VERSION


HARTREE_TO_KCAL_MOL = 627.5094740631


class DistributedEvalSampler(Sampler):
    """Partition evaluation data without DistributedSampler's padding copies."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class Metrics:
    NAMES = (
        "samples",
        "loss_sum",
        "energy_loss_sum",
        "qm_force_loss_sum",
        "mm_force_loss_sum",
        "energy_abs_sum",
        "energy_sq_sum",
        "qm_force_abs_sum",
        "qm_force_sq_sum",
        "qm_force_count",
        "mm_force_abs_sum",
        "mm_force_sq_sum",
        "mm_force_count",
    )

    def __init__(self, device: torch.device):
        self.values = torch.zeros(len(self.NAMES), device=device, dtype=torch.float64)

    def add(self, loss, parts, energy_error, qm_error, mm_error):
        values = {
            "samples": 1.0,
            "loss_sum": loss.detach(),
            "energy_loss_sum": parts["energy"].detach(),
            "qm_force_loss_sum": parts["qm_force"].detach(),
            "mm_force_loss_sum": parts["mm_force"].detach(),
            "energy_abs_sum": energy_error.detach().abs(),
            "energy_sq_sum": energy_error.detach().square(),
            "qm_force_abs_sum": qm_error.detach().abs().sum(),
            "qm_force_sq_sum": qm_error.detach().square().sum(),
            "qm_force_count": qm_error.numel(),
            "mm_force_abs_sum": mm_error.detach().abs().sum(),
            "mm_force_sq_sum": mm_error.detach().square().sum(),
            "mm_force_count": mm_error.numel(),
        }
        self.values += torch.stack(
            [torch.as_tensor(values[name], device=self.values.device, dtype=torch.float64) for name in self.NAMES]
        )

    def reduce(self) -> Dict[str, float]:
        if dist.is_initialized():
            dist.all_reduce(self.values, op=dist.ReduceOp.SUM)
        value = {name: float(self.values[i].item()) for i, name in enumerate(self.NAMES)}
        samples = max(value["samples"], 1.0)
        result = {
            "loss": value["loss_sum"] / samples,
            "energy_loss": value["energy_loss_sum"] / samples,
            "qm_force_loss": value["qm_force_loss_sum"] / samples,
            "mm_force_loss": value["mm_force_loss_sum"] / samples,
            "energy_mae_kcal_mol": value["energy_abs_sum"] / samples * HARTREE_TO_KCAL_MOL,
            "energy_rmse_kcal_mol": math.sqrt(value["energy_sq_sum"] / samples) * HARTREE_TO_KCAL_MOL,
            "qm_force_mae_kcal_mol_a": value["qm_force_abs_sum"] / max(value["qm_force_count"], 1.0) * HARTREE_TO_KCAL_MOL,
            "qm_force_rmse_kcal_mol_a": math.sqrt(value["qm_force_sq_sum"] / max(value["qm_force_count"], 1.0)) * HARTREE_TO_KCAL_MOL,
            "mm_force_mae_kcal_mol_a": value["mm_force_abs_sum"] / max(value["mm_force_count"], 1.0) * HARTREE_TO_KCAL_MOL,
            "mm_force_rmse_kcal_mol_a": math.sqrt(value["mm_force_sq_sum"] / max(value["mm_force_count"], 1.0)) * HARTREE_TO_KCAL_MOL,
        }
        return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--resume", type=Path, help="Resume from a last.pt checkpoint")
    parser.add_argument("--run-dir", type=Path, help="Override output_dir/run_name")
    parser.add_argument("--smoke-test", action="store_true", help="Use two train and two validation samples for one epoch")
    return parser.parse_args()


def initialize_parallel(use_gpu):
    """Use HydraGNN's Frontier-aware DDP setup inside a launched job.

    Keeping the MPI/deepspeed import out of a plain single-process shell makes
    ``--help`` and CPU smoke tests usable on login/service nodes as well.
    """
    launched = any(
        os.getenv(name) is not None
        for name in (
            "OMPI_COMM_WORLD_RANK",
            "SLURM_PROCID",
            "PALS_RANKID",
            "PMI_RANK",
        )
    )
    if launched:
        from hydragnn.utils.distributed import get_device, setup_ddp

        world_size, rank = setup_ddp()
        return world_size, rank, get_device(use_gpu=use_gpu)
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    return 1, 0, device


def single_sample_collate(batch):
    if len(batch) != 1:
        raise ValueError("This full-MM force model requires batch_size=1")
    return batch[0]


def seed_all(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def move_sample(sample: Dict, device: torch.device) -> Dict:
    result = dict(sample)
    for name in ("qm_z", "qm_pos", "mm_pos", "mm_charge", "energy", "qm_force", "mm_force"):
        result[name] = sample[name].to(device, non_blocking=True)
    result["qm_pos"] = result["qm_pos"].detach().requires_grad_(True)
    result["mm_pos"] = result["mm_pos"].detach().requires_grad_(True)
    return result


def model_dtype(training_config):
    precision = str(training_config.get("precision", "float64")).lower()
    if precision in ("float64", "fp64", "double"):
        return torch.float64
    if precision in ("float32", "fp32", "single"):
        return torch.float32
    raise ValueError("Unsupported training precision: {}".format(precision))


def objective(prediction, sample, force_qm, force_mm, loss_config):
    scales = loss_config["scales"]
    weights = loss_config["weights"]
    parts = {
        "energy": F.l1_loss(
            prediction / float(scales["energy_hartree"]),
            sample["energy"] / float(scales["energy_hartree"]),
        ),
        "qm_force": F.l1_loss(
            force_qm / float(scales["qm_force_hartree_per_angstrom"]),
            sample["qm_force"] / float(scales["qm_force_hartree_per_angstrom"]),
        ),
        "mm_force": F.l1_loss(
            force_mm / float(scales["mm_force_hartree_per_angstrom"]),
            sample["mm_force"] / float(scales["mm_force_hartree_per_angstrom"]),
        ),
    }
    loss = sum(float(weights[name]) * parts[name] for name in parts)
    return loss, parts


def run_epoch(model, loader, device, loss_config, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    metrics = Metrics(device)
    for sample in loader:
        sample = move_sample(sample, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        energy = model(sample["qm_z"], sample["qm_pos"], sample["mm_pos"], sample["mm_charge"])
        force_qm, force_mm = torch.autograd.grad(
            energy,
            (sample["qm_pos"], sample["mm_pos"]),
            create_graph=training,
            retain_graph=training,
        )
        force_qm = -force_qm
        force_mm = -force_mm
        loss, parts = objective(energy, sample, force_qm, force_mm, loss_config)
        if training:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        metrics.add(
            loss,
            parts,
            energy - sample["energy"],
            force_qm - sample["qm_force"],
            force_mm - sample["mm_force"],
        )
    return metrics.reduce()


def raw_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, config):
    torch.save(
        {
            "epoch": epoch,
            "model": raw_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation_loss": best_loss,
            "config": config,
            "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        },
        path,
    )


def print_metrics(epoch, elapsed, train, validation):
    print(
        "epoch={:04d} time={:.1f}s train_loss={:.6g} val_loss={:.6g} "
        "val_E_MAE={:.4f} kcal/mol val_Fqm_MAE={:.4f} val_Fmm_MAE={:.4f} kcal/mol/A".format(
            epoch,
            elapsed,
            train["loss"],
            validation["loss"],
            validation["energy_mae_kcal_mol"],
            validation["qm_force_mae_kcal_mol_a"],
            validation["mm_force_mae_kcal_mol_a"],
        ),
        flush=True,
    )


def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    world_size, rank, device = initialize_parallel(
        use_gpu=bool(config["training"].get("use_gpu", True))
    )
    seed_all(int(config["training"]["seed"]), rank)

    base_dir = config_path.parent
    dataset_root = (base_dir / config["data"]["dataset_root"]).resolve()
    split_dir = (base_dir / config["data"]["split_dir"]).resolve()
    split_files = {
        split: read_split_files(dataset_root, split_dir, split)
        for split in ("train", "val", "test")
    }
    if args.smoke_test:
        split_files["train"] = split_files["train"][:2]
        split_files["val"] = split_files["val"][:2]
        split_files["test"] = split_files["test"][:2]
        config["training"]["epochs"] = 1

    bias_setting = config["model"].get("energy_bias_init", "train_mean")
    bias_tensor = torch.zeros(1, device=device, dtype=torch.float64)
    if rank == 0:
        bias_tensor[0] = (
            MBE3Dataset.energy_mean(split_files["train"])
            if bias_setting == "train_mean"
            else float(bias_setting)
        )
    if dist.is_initialized():
        dist.broadcast(bias_tensor, src=0)
    energy_bias = float(bias_tensor.item())

    datasets = {split: MBE3Dataset(files) for split, files in split_files.items()}
    train_sampler = (
        DistributedSampler(
            datasets["train"], num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(config["training"]["seed"])
        )
        if world_size > 1
        else None
    )
    eval_samplers = {
        split: DistributedEvalSampler(datasets[split], rank, world_size)
        if world_size > 1 else None
        for split in ("val", "test")
    }
    workers = int(config["data"].get("num_workers", 0))
    loader_kwargs = dict(
        batch_size=1,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        collate_fn=single_sample_collate,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"], sampler=train_sampler,
            shuffle=train_sampler is None, **loader_kwargs
        ),
        "val": DataLoader(datasets["val"], sampler=eval_samplers["val"], **loader_kwargs),
        "test": DataLoader(datasets["test"], sampler=eval_samplers["test"], **loader_kwargs),
    }

    parameter_dtype = model_dtype(config["training"])
    model = MBE3QMMMModel(config["model"], energy_bias=energy_bias).to(
        device=device, dtype=parameter_dtype
    )
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
        min_lr=float(config["training"]["minimum_learning_rate"]),
    )

    run_dir = args.run_dir.resolve() if args.run_dir else (
        base_dir / config["output"]["output_dir"] / config["output"]["run_name"]
    ).resolve()
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = dict(config)
        resolved["resolved_dataset_root"] = str(dataset_root)
        resolved["resolved_split_dir"] = str(split_dir)
        resolved["resolved_energy_bias_init_hartree"] = energy_bias
        resolved["split_sizes"] = {name: len(files) for name, files in split_files.items()}
        (run_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
        print(
            "Training total MBE(3): {} train / {} val / {} test; {} rank(s); bias {:.12f} Ha".format(
                len(datasets["train"]), len(datasets["val"]), len(datasets["test"]),
                world_size, energy_bias
            ), flush=True
        )

    start_epoch = 1
    best_loss = float("inf")
    epochs_without_improvement = 0
    resume_path = args.resume
    if resume_path is not None:
        checkpoint = torch.load(str(resume_path), map_location=device)
        checkpoint_version = int(checkpoint.get("model_architecture_version", 1))
        if checkpoint_version != MODEL_ARCHITECTURE_VERSION:
            raise ValueError(
                "Checkpoint architecture version {} is incompatible with version {}. "
                "The corrected equivariant field embedding and bias-free PaiNN vector "
                "maps require a fresh training run.".format(
                    checkpoint_version, MODEL_ARCHITECTURE_VERSION
                )
            )
        raw_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])

    csv_path = run_dir / "metrics.csv"
    fieldnames = ["epoch", "learning_rate"] + [
        "{}_{}".format(split, metric)
        for split in ("train", "val")
        for metric in (
            "loss", "energy_loss", "qm_force_loss", "mm_force_loss",
            "energy_mae_kcal_mol", "energy_rmse_kcal_mol",
            "qm_force_mae_kcal_mol_a", "qm_force_rmse_kcal_mol_a",
            "mm_force_mae_kcal_mol_a", "mm_force_rmse_kcal_mol_a",
        )
    ]
    if rank == 0 and (not csv_path.exists() or start_epoch == 1):
        with csv_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.time()
        train_metrics = run_epoch(
            model, loaders["train"], device, config["loss"], optimizer,
            config["training"].get("gradient_clip_norm")
        )
        # Bypass the DDP wrapper during evaluation. Parameters are already
        # synchronized, and this permits exact unpadded eval partitions with
        # unequal sample counts across ranks.
        validation_metrics = run_epoch(
            raw_model(model), loaders["val"], device, config["loss"]
        )
        validation_loss = validation_metrics["loss"]
        if not math.isfinite(validation_loss):
            raise RuntimeError(
                "Non-finite validation loss at epoch {}; inspect last.pt/log output".format(epoch)
            )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - float(config["training"].get("minimum_improvement", 0.0))
        if improved:
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if rank == 0:
            row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
            for split, metrics in (("train", train_metrics), ("val", validation_metrics)):
                row.update({"{}_{}".format(split, key): value for key, value in metrics.items()})
            with csv_path.open("a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_loss, config)
            if improved:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_loss, config)
            print_metrics(epoch, time.time() - started, train_metrics, validation_metrics)

        if epochs_without_improvement >= int(config["training"]["early_stopping_patience"]):
            if rank == 0:
                print("Early stopping at epoch {}".format(epoch), flush=True)
            break

    if dist.is_initialized():
        dist.barrier()
    best_checkpoint = torch.load(str(run_dir / "best.pt"), map_location=device)
    raw_model(model).load_state_dict(best_checkpoint["model"])
    test_metrics = run_epoch(
        raw_model(model), loaders["test"], device, config["loss"]
    )
    if rank == 0:
        summary = {
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_validation_loss": float(best_checkpoint["best_validation_loss"]),
            "test": test_metrics,
        }
        (run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("Best-checkpoint test metrics: {}".format(json.dumps(test_metrics)), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
