"""Validate source snapshots and explicitly cached inputs before computation."""
from pathlib import Path
import hashlib
import json

from sberindex.paths import ROOT
CACHED=[
    'rolling_predictions.parquet','chronos2_rolling.csv','hgb_12m_predictions.csv',
    'category_forecast_comparison.csv','change_robustness.csv',
    'geographic_extension_predictions.parquet','geographic_extension_protocol.json',
    'geographic_extension_regions.csv','geographic_extension_summary.csv',
    'asof_cohort_predictions.parquet','asof_cohort_protocol.json',
    'asof_foundation_predictions.parquet','asof_foundation_protocol.json',
    'foundation_seasonal_predictions.parquet','foundation_seasonal_protocol.json',
    'foundation_seasonal_runtime.csv',
    'foundation_delay_predictions.parquet','foundation_delay_protocol.json',
    'foundation_delay_runtime.csv',
    'operational_early_predictions.parquet','operational_early_protocol.json',
    'operational_early_runtime.csv',
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root=ROOT, refresh=True):
    root=Path(root)
    sources=json.loads((root/'data_sources.json').read_text())
    for entry in sources['sources']:
        path=root/entry['path']
        if not path.exists() or sha(path)!=entry['sha256']:
            raise ValueError(f'Source snapshot missing or changed: {entry["path"]}')
    if refresh:
        manifest=json.loads((root/'refresh_inputs.json').read_text())
        if sha(root/'config.json')!=manifest['config_sha256']:
            raise ValueError('Config changed since saved forecasts were registered. Run full training or restore the recorded config.')
        for name,digest in manifest['cached_sha256'].items():
            path=root/'results'/name
            if not path.exists() or sha(path)!=digest:
                raise ValueError(f'Cached refresh input missing or changed: {name}')


def register(root=ROOT, provenance='inventory of existing cached outputs; original training code snapshot unavailable'):
    root=Path(root)
    manifest={'config_sha256':sha(root/'config.json'),
              'cached_sha256':{name:sha(root/'results'/name) for name in CACHED},
              'provenance':provenance,
              'limits':'Checksums detect changes; they do not prove correctness, historical availability or independent validation.'}
    (root/'refresh_inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    check()
    print('Source snapshots, config and cached refresh inputs match registered hashes.')
