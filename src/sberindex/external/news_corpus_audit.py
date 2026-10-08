"""Controlled corpus ablation; fixed 2023 training and reused 2024 evaluation.

This inexpensive HGB audit is refitted even in refresh mode. Its forecasts are
not inputs to the primary model. National announcements give only nine unique
training time points, regardless of the number of municipalities.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sberindex.external.news_ablation import known_news

from sberindex.paths import ROOT
def load_events():
    return pd.read_csv(ROOT / 'data/external/cbr_rate_decisions_2023_2024.csv',
                       parse_dates=['published_date', 'available_from'])


def select_training_cohort(panel):
    """Freeze eligibility using the completed training year only."""
    assert list(panel.columns[:12]) == [f'2023-{month:02d}' for month in range(1, 13)]
    return panel.loc[panel.iloc[:, :12].notna().all(axis=1)].sort_index()


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    raw = pd.read_parquet(ROOT / 'data/consumption.parquet')
    panel = select_training_cohort(raw[raw.category == 'Все категории'].pivot(
        index='territory_id', columns='date', values='value'))
    values = panel.to_numpy(float)
    logs = np.log(values)
    months = pd.date_range('2023-01-01', periods=24, freq='MS')
    events = load_events()
    corpora = {'hikes_only': events[events.delta_bps != 0], 'all_decisions': events}
    features = {name: [] for name in ['without_news', *corpora]}
    feature_rows = []
    for i in range(2, 23):
        base = np.column_stack([logs[:, i], logs[:, i]-logs[:, i-1],
            logs[:, i-1]-logs[:, i-2],
            np.full(len(panel), np.sin(2*np.pi*((i+1)%12)/12)),
            np.full(len(panel), np.cos(2*np.pi*((i+1)%12)/12))])
        features['without_news'].append(base)
        for name, corpus in corpora.items():
            vector = known_news(months[i], corpus)
            features[name].append(np.column_stack([base, np.tile(vector, (len(panel), 1))]))
            feature_rows.append((months[i].strftime('%Y-%m'), name, *vector))
    kwargs = dict(max_iter=config['hgb_max_iter'], max_leaf_nodes=config['hgb_max_leaf_nodes'],
        learning_rate=config['hgb_learning_rate'], min_samples_leaf=config['hgb_min_samples_leaf'],
        l2_regularization=config['hgb_l2_regularization'], random_state=config['random_seed'])
    target = np.concatenate([logs[:, i+1]-logs[:, i] for i in range(2, 11)])
    rows = []
    coverage = []
    for i in range(12, 23):
        history_ok = np.isfinite(logs[:, i-2:i+1]).all(axis=1)
        actual_ok = np.isfinite(values[:, i+1])
        coverage.append(dict(origin=str(months[i].to_period('M')),
            target=str(months[i+1].to_period('M')), eligible_ids=len(panel),
            missing_history=int((~history_ok).sum()), missing_target=int((~actual_ok).sum()),
            scored_ids=int((history_ok & actual_ok).sum()),
            unscored_ids=int((~(history_ok & actual_ok)).sum())))
    fitted = {}
    for name, blocks in features.items():
        model = HistGradientBoostingRegressor(**kwargs).fit(np.concatenate(blocks[:9]), target)
        fitted[name] = int(model.n_iter_)
        for i in range(12, 23):
            history_ok = np.isfinite(logs[:, i-2:i+1]).all(axis=1)
            scored = history_ok & np.isfinite(values[:, i+1])
            # HGB can accept NaNs, but the current spending anchor cannot.
            # Use a common complete three-month input window in all arms.
            predicted = np.exp(logs[scored, i]+np.clip(model.predict(blocks[i-2][scored]), -.5, .5))
            rows.extend((int(city), months[i].strftime('%Y-%m'), months[i+1].strftime('%Y-%m'),
                         name, float(y), float(p)) for city,y,p in zip(panel.index[scored],values[scored,i+1],predicted))
    predictions = pd.DataFrame(rows, columns=['territory_id','origin','target','model','actual','predicted'])
    predictions['period'] = np.where(predictions.target < '2024-07', 'early_feb_jun', 'late_jul_dec')
    predictions['ae'] = abs(predictions.actual-predictions.predicted)
    summary = []
    for (period,name), group in predictions.groupby(['period','model']):
        summary.append(dict(period=period,model=name,MAE=group.ae.mean(),R2=r2_score(group.actual,group.predicted),
            municipalities=group.territory_id.nunique(),dates=group.target.nunique()))
    out = ROOT / 'results'
    predictions.to_parquet(out/'news_corpus_predictions.parquet',index=False)
    pd.DataFrame(coverage).to_csv(out/'news_corpus_coverage.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'news_corpus_summary.csv',index=False)
    predictions.groupby(['target','model'],as_index=False).ae.mean().rename(columns={'ae':'MAE'}).to_csv(
        out/'news_corpus_monthly.csv',index=False)
    pd.DataFrame(feature_rows,columns=['origin','corpus','announced_rate_pct','delta_90d_pp','announcements_90d']).to_csv(
        out/'news_corpus_features.csv',index=False)
    protocol = dict(training_origins='2023-03–2023-11', training_targets='2023-04–2023-12',
        unique_training_origins=9, training_rows=len(target), events=len(events), holds=int((events.delta_bps==0).sum()),
        cohort_selection='All IDs with 12 observed months in 2023; no 2024 completeness filter',
        eligible_ids=len(panel), training_ids=panel.index.astype(int).tolist(),
        evaluation_missingness='Same three observed input months and observed target in all arms; no imputation; per-target coverage saved',
        available_from_rule='publication + 1 day; announcement information, not effective date',
        national_scope='Same news vector for all municipalities; municipality count is not independent news sample size',
        historical_rate_initial_pct=7.5, model_kwargs=kwargs, fitted_iterations=fitted,
        early_stopping='Inherited sklearn auto, random internal split within 2023 only; no temporal model selection guarantee',
        selection='No selection or primary-model replacement; all three arms reported on early and late reused dates',
        limitations='Structured policy releases only, not comprehensive news/NLP. No causal inference. No independent test. Future target-month news excluded.')
    (out/'news_corpus_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
