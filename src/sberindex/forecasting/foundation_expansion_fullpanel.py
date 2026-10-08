"""Resource bounded full-panel inference and fixed 2023-only Bolt pilot."""

import hashlib
import json
import resource
import time

import numpy as np
import pandas as pd

from sberindex.forecasting.foundation_expansion_review import (
    KEYS,
    OUT,
    finetune_split,
    metrics,
    panel_data,
)
from sberindex.paths import ROOT


def run(stage):
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2023)
    panel, profile = panel_data()
    full = pd.read_parquet(ROOT / "reports/full_cohort_review/predictions.parquet")
    base = full[full.model.eq("seasonal_pooled")].copy()
    cfg = json.loads((ROOT / "configs/foundation_covariate_review.json").read_text())
    records = []
    runtime = []
    if stage.startswith("chronos2"):
        from chronos import Chronos2Pipeline

        pipe = Chronos2Pipeline.from_pretrained(
            cfg["chronos2_model"], revision=cfg["chronos2_revision"], device_map="cpu"
        )
    elif stage.startswith("bolt_finetune"):
        from chronos import ChronosBoltPipeline

        pipe = ChronosBoltPipeline.from_pretrained(
            cfg["bolt_model"], revision=cfg["bolt_revision"], device_map="cpu"
        )
        valid = (np.isfinite(panel.iloc[:, :12]) & (panel.iloc[:, :12] > 0)).all(axis=1)
        train, val = finetune_split(panel.loc[valid].to_numpy("float32"))
        model = pipe.model
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        tick = time.monotonic()
        losses = []
        sampled_rows = set()
        epoch1 = stage == "bolt_finetune_epoch1"
        batch_size = 128 if epoch1 else 16
        if epoch1:
            permutation = torch.randperm(len(train))
            batches = list(permutation.split(batch_size))
        else:
            batches = [torch.randperm(len(train))[:batch_size] for _ in range(20)]
        for step, rows in enumerate(batches):
            sampled_rows.update(rows.tolist())
            history = torch.from_numpy(train[rows, :6])
            target = torch.from_numpy(train[rows, 6:9])
            opt.zero_grad()
            loss = model(context=history, target=target).loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3 > 10:
                raise MemoryError("10GiB model-process limit")
            print("bolt step", step, float(loss.detach()), flush=True)
        model.eval()
        validation_count = len(val) if epoch1 else 64
        validation_parts = []
        with torch.inference_mode():
            for offset in range(0, validation_count, batch_size):
                chunk = val[offset : min(offset + batch_size, validation_count)]
                value = float(
                    model(
                        context=torch.from_numpy(chunk[:, :9]),
                        target=torch.from_numpy(chunk[:, 9:12]),
                    ).loss
                )
                validation_parts.append((len(chunk), value))
        validation_loss = sum(n * x for n, x in validation_parts) / validation_count
        path = OUT / (
            "bolt_finetuned_epoch1_checkpoint"
            if epoch1
            else "bolt_finetuned_checkpoint"
        )
        model.save_pretrained(path)
        checkpoint_manifest = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.iterdir()
            if p.is_file()
        }
        (
            OUT / ("bolt_epoch1_training.json" if epoch1 else "bolt_training.json")
        ).write_text(
            json.dumps(
                {
                    "train_end": "2023-09",
                    "validation_start": "2023-10",
                    "validation_end": "2023-12",
                    "eligible_training_municipalities": len(train),
                    "sampled_training_municipalities": len(sampled_rows),
                    "steps": len(batches),
                    "fixed_epochs": 1 if epoch1 else None,
                    "batch_size": batch_size,
                    "learning_rate": 1e-5,
                    "seed": 2023,
                    "train_context": "2023-01..2023-06",
                    "train_targets": "2023-07..2023-09",
                    "validation_municipalities": validation_count,
                    "validation_loss": validation_loss,
                    "training_losses": losses,
                    "seconds": time.monotonic() - tick,
                    "early_stopping": False,
                    "checkpoint_fixed_before_2024_inference": True,
                    "checkpoint_sha256": checkpoint_manifest,
                },
                indent=2,
            )
        )
        frozen = pd.read_csv(
            ROOT / "reports/presentation_comparison_cases/common_pairs.csv"
        )
        base = frozen.merge(base, on=KEYS, validate="one_to_one")
    else:
        raise ValueError(stage)
    for origin, g in base.groupby("origin", sort=True):
        tick = time.monotonic()
        ids = np.sort(g.territory_id.unique())
        t = pd.Period(origin, "M").ordinal - pd.Period("2023-01", "M").ordinal
        h = int(g.horizon.max())
        history = panel.loc[ids].iloc[:, : t + 1].to_numpy("float32")
        if stage.startswith("chronos2"):
            tasks = []
            for x in history:
                task = {"target": x}
                if stage == "chronos2_profile_full":
                    task.update(
                        past_covariates={
                            "profile": profile[np.arange(t + 1) % 12].astype("float32")
                        },
                        future_covariates={
                            "profile": profile[(t + np.arange(1, h + 1)) % 12].astype(
                                "float32"
                            )
                        },
                    )
                tasks.append(task)
            with torch.inference_mode():
                pred = pipe.predict(
                    tasks, prediction_length=h, batch_size=96, cross_learning=False
                )
            q = pipe.quantiles.index(0.5)
            forecast = np.stack([p[0, q, :].cpu().numpy() for p in pred])
        else:
            parts = []
            with torch.inference_mode():
                for start in range(0, len(history), 32):
                    parts.append(
                        pipe.predict(
                            torch.from_numpy(history[start : start + 32]),
                            prediction_length=h,
                        )[:, 4, :]
                        .cpu()
                        .numpy()
                    )
            forecast = np.concatenate(parts)
        assert np.isfinite(forecast).all()
        for row in g.itertuples():
            i = np.searchsorted(ids, row.territory_id)
            r = {k: getattr(row, k) for k in KEYS}
            r.update(
                model=stage,
                category="Все категории",
                actual=float(row.actual),
                predicted=float(max(0, forecast[i, row.horizon - 1])),
                year_ago=float(row.year_ago),
            )
            records.append(r)
        runtime.append(
            {
                "stage": stage,
                "origin": origin,
                "municipalities": len(ids),
                "pairs": len(g),
                "peak_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**3,
                "seconds": time.monotonic() - tick,
            }
        )
        print(runtime[-1], flush=True)
        pd.DataFrame(records).to_parquet(
            OUT / f"{stage}_predictions.parquet", index=False
        )
        pd.DataFrame(runtime).to_csv(OUT / f"{stage}_runtime.csv", index=False)
    pd.DataFrame(
        [
            dict(model=stage, horizon=int(h), **metrics(g))
            for h, g in pd.DataFrame(records).groupby("horizon")
        ]
    ).to_csv(OUT / f"{stage}_summary.csv", index=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=[
            "chronos2_univariate_full",
            "chronos2_profile_full",
            "bolt_finetune",
            "bolt_finetune_epoch1",
        ],
        required=True,
    )
    run(p.parse_args().stage)
