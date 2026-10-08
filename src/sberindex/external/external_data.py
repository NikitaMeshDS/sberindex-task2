"""Version-aware Sber municipal lookup and retrospective regional Rosstat CPI.

The 2026 CPI vintage is used for interpretation, never as a certified historical
forecast feature. Territory records are joined only for year_from <= year < year_to.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
EXT = ROOT / "data/external"


def normalize_name(value):
    return re.sub(r"[^а-я0-9]", "", str(value).lower().replace("ё", "е").replace("г.", ""))


def main():
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    versions = pd.read_excel(EXT / "sber_municipal_versions_20241025.xlsx")
    lookup = pd.concat([versions[(versions.year_from <= year) & (versions.year_to > year)]
                        .assign(year=year) for year in (2023, 2024)], ignore_index=True)
    assert not lookup.duplicated(["territory_id", "year"]).any()
    lookup.to_csv(ROOT / "results/municipal_lookup.csv", index=False)
    keys = ["territory_id", "year", "region_code", "region_name", "oktmo",
            "municipal_district_name_short", "municipal_district_name", "year_from", "year_to"]
    joined = raw.assign(year=raw.date.str[:4].astype(int)).merge(
        lookup[keys], on=["territory_id", "year"], how="left", validate="many_to_one", indicator=True)
    assert len(joined) == len(raw)
    assert joined._merge.eq("both").all(), joined.loc[joined._merge != "both", "territory_id"].unique()
    joined = joined.drop(columns="_merge")

    rows = []
    with pd.ExcelFile(EXT / "rosstat_ipc_RF_fo_sub_08-2026.xlsx") as book:
        for year in (2022, 2023, 2024):
            for month in range(1, 13):
                sheet = pd.read_excel(book, sheet_name=f"{month:02d}({year})", header=None)
                for code, name, all_cpi, food, nonfood, services in sheet.iloc[4:, :6].itertuples(index=False, name=None):
                    if isinstance(code, (int, float)) and pd.notna(code) and pd.notna(all_cpi):
                        rows.append((int(code), name, f"{year}-{month:02d}", float(all_cpi),
                                     float(food), float(nonfood), float(services)))
    cpi = pd.DataFrame(rows, columns=["rosstat_region_code", "rosstat_region_name", "date",
                                      "cpi_all_mom", "cpi_food_mom", "cpi_nonfood_mom", "cpi_services_mom"])
    assert not cpi.duplicated(["rosstat_region_code", "date"]).any()
    reference = cpi[cpi.date == "2024-01"].set_index("rosstat_region_code")
    name_to_code = {normalize_name(name): code for code, name in reference.rosstat_region_name.items()}
    # Explicit source-verified equivalences, including exclusion of autonomous
    # districts from the corresponding parent-oblast CPI to avoid double counting.
    overrides = {
        "Архангельская область": 11001000,
        "Тюменская область": 71001000,
        "Ненецкий автономный округ": 11100000,
        "Ханты-Мансийский автономный округ — Югра": 71100000,
        "Ямало-Ненецкий автономный округ": 71140000,
        "Кемеровская область": 32000000,
        "Еврейская автономная область": 99000000,
        "Чукотский автономный округ": 77000000,
    }
    crosswalk = []
    for code, name in lookup[["region_code", "region_name"]].drop_duplicates().itertuples(index=False, name=None):
        target = overrides.get(name, name_to_code.get(normalize_name(name)))
        assert target is not None, name
        crosswalk.append((code, name, target, reference.loc[target, "rosstat_region_name"],
                          "explicit_equivalence" if name in overrides else "exact_normalized_name"))
    crosswalk = pd.DataFrame(crosswalk, columns=["region_code", "region_name", "rosstat_region_code",
                                                "rosstat_region_name", "join_rule"])
    assert crosswalk.region_code.is_unique and crosswalk.rosstat_region_code.is_unique
    crosswalk.to_csv(ROOT / "results/region_crosswalk.csv", index=False)
    cpi = cpi.sort_values(["rosstat_region_code", "date"])
    for kind in ("all", "food"):
        cpi[f"index_{kind}"] = (cpi[f"cpi_{kind}_mom"] / 100).groupby(cpi.rosstat_region_code).cumprod()
        cpi[f"factor_{kind}_yoy"] = cpi[f"index_{kind}"] / cpi.groupby("rosstat_region_code")[f"index_{kind}"].shift(12)
        base = cpi[cpi.date == "2022-12"].set_index("rosstat_region_code")[f"index_{kind}"]
        cpi[f"factor_{kind}_dec2022"] = cpi[f"index_{kind}"] / cpi.rosstat_region_code.map(base)
    cpi["vintage"] = "2026-09-11"
    cpi["usage"] = "retrospective_only_not_historical_vintage"
    cpi.to_csv(ROOT / "results/rosstat_cpi_monthly.csv", index=False)
    joined = joined.merge(crosswalk[["region_code", "rosstat_region_code"]], on="region_code", validate="many_to_one")
    joined = joined.merge(cpi.drop(columns="rosstat_region_name"), on=["rosstat_region_code", "date"],
                          how="left", validate="many_to_one")
    assert joined.cpi_all_mom.notna().all()
    joined["real_value_dec2022_prices"] = np.nan
    for category, kind in (("Все категории", "all"), ("Продовольствие", "food")):
        mask = joined.category.eq(category)
        joined.loc[mask, "real_value_dec2022_prices"] = (
            joined.loc[mask, "value"] / joined.loc[mask, f"factor_{kind}_dec2022"])
    joined.to_parquet(ROOT / "results/consumption_with_geography_cpi.parquet", index=False)
    audit = {"input_rows": len(raw), "joined_rows": len(joined), "matched_share": 1.0,
             "municipal_ids": int(raw.territory_id.nunique()), "regions": int(joined.region_code.nunique()),
             "lookup_rule": "year_from <= observation_year < year_to",
             "cpi_join": "explicit region crosswalk plus observation month",
             "cpi_vintage": "2026-09-11", "use_in_forecast": False,
             "municipal_dictionary_coverage_end": "2024-01-01",
             "caveat": "Intra-year 2024 boundary changes are not represented; dictionary supports retrospective labels, not proof of stable measurement."}
    (ROOT / "results/external_join_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
