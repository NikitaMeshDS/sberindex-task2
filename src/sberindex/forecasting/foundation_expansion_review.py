"""Pinned foundation expansion; causal inputs and exact cached evaluation keys."""

import argparse
import json
import resource
import time

import numpy as np
import pandas as pd

from sberindex.forecasting.foundation_covariate_review import metrics
from sberindex.paths import ROOT

OUT = ROOT / "reports/foundation_expansion_review"
EXT = ROOT / "data/external/foundation_expansion_review"
KEYS = ["territory_id", "origin", "target", "horizon"]


def ratio_history(values, origin, profile):
    """2023 level times frozen calendar profile; observations stop at origin."""
    values = np.asarray(values, float)
    profile = np.asarray(profile, float)
    if origin < 11 or profile.shape != (12,):
        raise ValueError("complete 2023 and 12-month profile required")
    scale = np.mean(values[:12] / profile)
    return (
        values[: origin + 1] / (scale * profile[np.arange(origin + 1) % 12])
    ).astype("float32"), scale


def matched(frame, keys):
    if frame.duplicated(KEYS).any() or keys.duplicated(KEYS).any():
        raise ValueError("duplicate evaluation key")
    result = keys.merge(
        frame, on=KEYS, how="left", validate="one_to_one", indicator=True
    )
    if not result._merge.eq("both").all():
        raise ValueError("missing frozen evaluation key")
    return result.drop(columns="_merge")


def finetune_split(values):
    values = np.asarray(values)
    return values[:, :9].copy(), values[:, :12].copy()


def panel_data():
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = raw[raw.category.eq("Все категории")].pivot(
        index="territory_id", columns="date", values="value"
    )
    panel = panel.reindex(
        columns=pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    ).sort_index()
    valid = (np.isfinite(panel.iloc[:, :12]) & (panel.iloc[:, :12] > 0)).all(axis=1)
    sums = panel.loc[valid].iloc[:, :12].sum().to_numpy(float)
    return panel, sums / sums.mean()


def backend(name, cfg):
    import torch
    from huggingface_hub import snapshot_download

    spec = cfg["models"][name]
    cache = snapshot_download(spec["model_id"], revision=spec["revision"])
    if name == "timesfm25":
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            cache, torch_compile=False
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=32,
                max_horizon=12,
                per_core_batch_size=32,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )
        return lambda history, h: np.asarray(
            model.forecast(horizon=h, inputs=list(history))[0]
        )
    if name == "tirex":
        from tirex import load_model

        model = load_model(
            spec["model_id"],
            device="cpu",
            backend="torch",
            hf_kwargs={"revision": spec["revision"]},
        )

        def predict(history, h):
            q, _ = model.forecast(
                torch.as_tensor(history),
                prediction_length=h,
                batch_size=32,
                output_type="numpy",
                dynamic_padding=True,
            )
            return np.asarray(q)[:, :, 4]

        return predict
    if name == "moirai2":
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        module = Moirai2Module.from_pretrained(cache)

        def predict(history, h):
            model = Moirai2Forecast(
                module=module,
                prediction_length=h,
                context_length=32,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            ds = ListDataset(
                [
                    {"start": pd.Period("2023-01", freq="M"), "target": x}
                    for x in history
                ],
                freq="M",
            )
            return np.stack(
                [
                    f.quantile(0.5)
                    for f in model.create_predictor(
                        batch_size=32, device="cpu"
                    ).predict(ds)
                ]
            )

        return predict
    raise ValueError(name)


def run_model(name):
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2023)
    cfg = json.loads((ROOT / "configs/foundation_expansion_review.json").read_text())
    panel, profile = panel_data()
    keys = pd.read_csv(ROOT / cfg["common_pairs"])
    base = pd.read_parquet(ROOT / "reports/growth_bridge_review/predictions.parquet")
    base = matched(base[base.model.eq("seasonal_pooled")], keys)
    tick = time.monotonic()
    predict = backend(name, cfg)
    runtime = [{"model": name, "stage": "load", "seconds": time.monotonic() - tick}]
    records = []
    for origin, g in keys.groupby("origin", sort=True):
        t = pd.Period(origin, "M").ordinal - pd.Period("2023-01", "M").ordinal
        ids = np.sort(g.territory_id.unique())
        hist = panel.loc[ids].iloc[:, : t + 1].to_numpy("float32")
        if not np.isfinite(hist).all() or not (hist > 0).all():
            raise ValueError("invalid frozen history")
        h = int(g.horizon.max())
        ratio_and_scale = [
            ratio_history(panel.loc[i].to_numpy(), t, profile) for i in ids
        ]
        ratios = np.stack([r[0] for r in ratio_and_scale])
        scales = np.array([r[1] for r in ratio_and_scale])
        for mode, inputs in [("zero_shot", hist), ("seasonal_ratio", ratios)]:
            tick = time.monotonic()
            with torch.inference_mode():
                forecast = predict(inputs, h)
            if forecast.shape != (len(ids), h) or not np.isfinite(forecast).all():
                raise ValueError(f"invalid shape {forecast.shape}")
            if mode == "seasonal_ratio":
                forecast *= scales[:, None] * profile[(t + np.arange(1, h + 1)) % 12]
            for row in g.itertuples():
                i = int(np.searchsorted(ids, row.territory_id))
                b = base[
                    (base.territory_id.eq(row.territory_id))
                    & base.origin.eq(origin)
                    & base.horizon.eq(row.horizon)
                ].iloc[0]
                predicted = float(max(0, forecast[i, row.horizon - 1]))
                records.append(
                    dict(
                        **{k: getattr(row, k) for k in KEYS},
                        category="Все категории",
                        model=f"{name}_{mode}",
                        predicted=predicted,
                        actual=float(b.actual),
                        year_ago=float(b.year_ago),
                        origin_actual=float(b.origin_actual),
                    )
                )
                if mode == "zero_shot":
                    records.append(
                        dict(
                            **{k: getattr(row, k) for k in KEYS},
                            category="Все категории",
                            model=f"{name}_ensemble50",
                            predicted=(predicted + float(b.predicted)) / 2,
                            actual=float(b.actual),
                            year_ago=float(b.year_ago),
                            origin_actual=float(b.origin_actual),
                        )
                    )
            runtime.append(
                {
                    "model": name,
                    "mode": mode,
                    "stage": "inference",
                    "origin": origin,
                    "seconds": time.monotonic() - tick,
                    "peak_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024**3,
                    "tasks": len(hist),
                }
            )
            print(name, origin, mode, runtime[-1], flush=True)
        pd.DataFrame(records).to_parquet(
            OUT / f"{name}_predictions.parquet", index=False
        )
        pd.DataFrame(runtime).to_csv(OUT / f"{name}_runtime.csv", index=False)
    for _, f in pd.DataFrame(records).groupby("model"):
        matched(f, keys)


def report():
    cfg = json.loads((ROOT / "configs/foundation_expansion_review.json").read_text())
    keys = pd.read_csv(ROOT / cfg["common_pairs"])
    frames = []
    for file in [
        OUT / f"{name}_predictions.parquet"
        for name in [
            "timesfm25",
            "moirai2",
            "tirex",
            "bolt_finetune",
            "bolt_finetune_epoch1",
        ]
    ]:
        if not file.exists():
            continue
        for _, f in pd.read_parquet(file).groupby("model"):
            frames.append(matched(f, keys))
    for file, models in [
        (
            "reports/growth_bridge_review/predictions.parquet",
            [
                "seasonal_naive",
                "seasonal_pooled",
                "prophet_disabled",
                "prophet_pooled_profile",
                "prophet_yearly3",
                "hierarchy05",
                "hierarchy05_growth_bridge",
            ],
        ),
        (
            "reports/foundation_covariate_review/predictions.parquet",
            ["chronos2_univariate", "chronos2_profile", "bolt_base_univariate"],
        ),
    ]:
        f = pd.read_parquet(ROOT / file)
        if "category" in f:
            f = f[f.category.eq("Все категории")]
        for m in models:
            z = matched(f[f.model.eq(m)], keys)
            z["category"] = "Все категории"
            frames.append(z)
    direct = pd.read_parquet(ROOT / "reports/direct_horizon_review/predictions.parquet")
    direct = direct[
        direct.model.isin(
            ["direct_lags", "direct_lags__seasonal_fallback_not_trainable"]
        )
    ].copy()
    direct["status"] = np.where(
        direct.model.eq("direct_lags"),
        "trained_direct",
        "seasonal_fallback_not_trainable",
    )
    direct["model"] = "direct_lags"
    direct["category"] = "Все категории"
    frames.append(matched(direct, keys))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(OUT / "matched_predictions.parquet", index=False)
    summary = pd.DataFrame(
        [
            dict(model=m, horizon=int(h), **metrics(g))
            for (m, h), g in frame.groupby(["model", "horizon"])
        ]
    )
    reference = summary[summary.model.eq("seasonal_pooled")].set_index("horizon").MAE
    summary["skill_vs_seasonal_pooled"] = 1 - summary.MAE / summary.horizon.map(
        reference
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 7))
    pivot = summary.pivot(
        index="model", columns="horizon", values="skill_vs_seasonal_pooled"
    )
    pivot.plot.barh(ax=ax)
    ax.set_xlabel("Skill относительно seasonal pooled (выше лучше)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    fig.savefig(OUT / "comparison.svg")
    plt.close(fig)
    mae_matrix = summary.pivot(index="model", columns="horizon", values="MAE")
    markdown = (
        "| Модель | h1 MAE | h3 MAE | h6 MAE | h12 MAE |\n|---|---:|---:|---:|---:|\n"
    )
    markdown += "\n".join(
        "| "
        + str(m)
        + " | "
        + " | ".join(f"{row[h]:.2f}" for h in [1, 3, 6, 12])
        + " |"
        for m, row in mae_matrix.iterrows()
    )
    new_rows = summary[
        summary.model.str.startswith(("timesfm25_", "moirai2_", "tirex_"))
    ]
    outcome = (
        "Все девять фиксированных новых вариантов уступают seasonal pooled на каждом из четырех горизонтов. Остаточное отношение уменьшает ошибку zero-shot, но этого недостаточно для превосходства сезонного профиля. "
        if (new_rows.skill_vs_seasonal_pooled < 0).all()
        else "Результаты фиксированных вариантов приведены ниже. "
    )
    text = (
        "# Расширение foundation benchmark\n\n"
        + outcome
        + "Короткая месячная история 12–23 точки ограничивает выводы о переносимости моделей.\n\nСравнение проводится на точных замороженных ключах: h=1: 251 МО/12 целей, h=3: 251/10, h=6: 252/7, h=12: 252/1. Это ретроспективная исследовательская проверка, не независимый holdout.\n\nНовые checkpoints выпущены после 2024: отсутствие данных SberIndex в предобучении не подтверждено. Возможная контаминация неизвестна.\n\nПрофиль и масштаб остаточного отношения оцениваются только по 2023; вход истории заканчивается origin. Seasonal ratio восстанавливается с фиксированным профилем 2023. Ансамбль фиксирован 50/50 с seasonal pooled; параметры не подбирались по 2024. TiRex использует документированный dynamic_padding=True для короткой истории; это ограничение сопоставимости. PyPI1.4.2 отклоняет этот аргумент; успешно применена установка точной upstream Git revision, ошибка сохранена. TimesFM 2.5 используется через PyTorch на CPU; современная документация native MLX относится к 3.0. Direct HGB h12 в cached reference — явно не обучаемый seasonal fallback.\n\n"
        + markdown
        + "\n\nОфициальные источники: [TimesFM](https://github.com/google-research/timesfm), [Moirai 2](https://github.com/SalesforceAIResearch/uni2ts), [TiRex](https://github.com/NX-AI/tirex).\n"
    )
    if (OUT / "bolt_epoch1_training.json").exists():
        text += "\nChronos-Bolt: один фиксированный epoch, все 2075 муниципалитетов 2023,17 batches × 128 (последний неполный), lr 1e-5, seed 2023. Обучение Jan–Sep2023: context Jan–Jun иtargets Jul–Sep; validation Oct–Dec2023 для всех 2075. Без early stopping и 2024 validation; checkpoint_sha256 вbolt_epoch1_training.json фиксирован до 2024 inference.20-step pilot (300 МО) сохранен отдельно; бюджет одного epoch уточнен для выполнения требования полной панели, оба результата публикуются без выбора по 2024.\n\n[Официальная документация fine-tuning Bolt](https://auto.gluon.ai/1.4.0/tutorials/timeseries/forecasting-chronos.html#fine-tuning). Бюджетный цикл использует официальный quantile loss установленногоChronosBoltModule.forward(context,target).\n"
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["timesfm25", "moirai2", "tirex"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
    elif args.model:
        run_model(args.model)
