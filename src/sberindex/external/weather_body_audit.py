"""Rebuild a frozen body sample and diagnose information loss in headlines.

Body regional matches are candidate entities, not validated event locations.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import pandas as pd
from sberindex.external.weather_news_audit import MONTHS,region_pattern

from sberindex.paths import ROOT
DATA=ROOT/'data/external/weather_body_sample'


def plain(text):
    text=re.sub(r'<script\b[^>]*>.*?</script>',' ',text,flags=re.S|re.I)
    text=re.sub(r'<style\b[^>]*>.*?</style>',' ',text,flags=re.S|re.I)
    return ' '.join(html.unescape(re.sub(r'<[^>]+>',' ',text)).split())


def parse_body(text):
    section=text.split('<h1>',1)[1].split('<!-- begin: BRED CRUMBS -->',1)[0]
    title,section=section.split('</h1>',1)
    date=re.search(r'<strong>(\d{1,2})\s+([а-я]+)\s+(20\d{2})\s*г</strong>',section)
    assert date,'Article publication date not found'
    published=f'{date[3]}-{MONTHS[date[2]]:02d}-{int(date[1]):02d}'
    body=plain(section[date.end():])
    assert len(body)>20, 'Empty or image-only body: manual inspection required'
    return plain(title),published,body


def coordinated_regions(text,regions):
    """Candidate mentions with explicit shared 'oblast'/'krai' nouns.

    Parentheses are omitted only for list parsing; direct full-text matches stay.
    No negation, event relation or temporal resolution is claimed here.
    """
    text=text.lower().replace('ё','е')
    matches={int(r.region_code) for r in regions.itertuples() if re.search(region_pattern(r.region_name),text)}
    clean=re.sub(r'\([^)]*\)',' ',text)
    adjective=r'[а-я-]+(?:ой|ая|ую|ых|ые|ом|ий|ие)'
    for noun,ending in [(r'област\w*','область'),(r'кра[йяеюх]\w*','край')]:
        pattern=rf'\b({adjective}(?:(?:\s*,\s*|\s+и\s+){adjective})*)\s+{noun}\b'
        for match in re.finditer(pattern,clean):
            words=re.findall(adjective,match[1])
            for r in regions.itertuples():
                if r.region_name.endswith(ending) and 'автономная' not in r.region_name:
                    stem=r.region_name.lower().split()[0][:-2]
                    if any(w.startswith(stem) for w in words):matches.add(int(r.region_code))
    return sorted(matches)


def main():
    sample=pd.read_csv(DATA/'sample.csv')
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert hashlib.sha256((DATA/'sample.csv').read_bytes()).hexdigest()==manifest['sample_sha256']
    assert {r['source_url'] for r in manifest['pages']}==set(sample.source_url)
    regions=pd.read_csv(ROOT/'results/region_crosswalk.csv')
    patterns={int(r.region_code):region_pattern(r.region_name) for r in regions.itertuples()}
    rows=[]
    for page in manifest['pages']:
        assert page['status']=='downloaded',page['source_url']
        path=ROOT/page['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==page['sha256']
        title,date,body=parse_body(path.read_text())
        row=sample[sample.source_url==page['source_url']].iloc[0]
        assert date==row.published_date,(date,row.published_date,page['source_url'])
        assert title==row.title,(title,row.title)
        title_regions=[c for c,p in patterns.items() if re.search(p,title.lower().replace('ё','е'))]
        body_regions=[c for c,p in patterns.items() if re.search(p,body.lower().replace('ё','е'))]
        rows.append(dict(source_url=page['source_url'],published_date=date,title=title,title_type=row.title_type,
            sampling_rule=row.sampling_rule,body=body,body_chars=len(body),title_regions=json.dumps(title_regions),
            body_regions=json.dumps(body_regions),new_body_regions=json.dumps(sorted(set(body_regions)-set(title_regions)))))
    frame=pd.DataFrame(rows).sort_values(['published_date','source_url'])
    frame.to_csv(ROOT/'results/weather_body_sample.csv',index=False)
    annotations=pd.read_csv(DATA/'annotations.csv').fillna('')
    reviewed=frame.merge(annotations,on='source_url',validate='one_to_one')
    assert len(reviewed)==len(frame)
    reviewed['title_predicts_weather']=reviewed.title_type.isin(['forecast_title','warning_title'])
    topic=reviewed.groupby(['title_type','body_class'],as_index=False).size()
    topic.to_csv(ROOT/'results/weather_body_topic_comparison.csv',index=False)
    reviewed.drop(columns='body').to_csv(ROOT/'results/weather_body_review.csv',index=False)
    warning=pd.read_csv(DATA/'warning_events.csv')
    geo=[];validity=[]
    for event in warning.itertuples():
        article=frame[frame.source_url==event.source_url].iloc[0]
        truth=set(json.loads(event.region_codes))
        for method,predicted in [('title',json.loads(article.title_regions)),('body_direct',json.loads(article.body_regions)),
                                 ('body_coordinated',coordinated_regions(article.body,regions))]:
            predicted=set(predicted)
            geo.append(dict(source_url=event.source_url,method=method,TP=len(predicted&truth),FP=len(predicted-truth),FN=len(truth-predicted),
                predicted_regions=json.dumps(sorted(predicted)),reference_regions=event.region_codes))
        for origin in pd.date_range('2023-01-01','2024-12-01',freq='MS'):
            end=origin+pd.offsets.MonthEnd(0)
            if pd.Timestamp(event.available_from)>end:continue
            for horizon in [1,3,6,12]:
                target=origin+pd.DateOffset(months=horizon)
                overlap=pd.Timestamp(event.event_start)<=target+pd.offsets.MonthEnd(0) and pd.Timestamp(event.event_end)>=target
                validity.append(dict(source_url=event.source_url,origin=origin.strftime('%Y-%m'),horizon=horizon,
                    target=target.strftime('%Y-%m'),forecast_period_overlaps_target=bool(overlap),expired_at_origin=pd.Timestamp(event.event_end)<=end))
                validity[-1]['publication_in_past_90d']=pd.Timestamp(event.available_from)>end-pd.Timedelta(days=90)
    pd.DataFrame(geo).to_csv(ROOT/'results/weather_body_geography_comparison.csv',index=False)
    pd.DataFrame(validity).to_csv(ROOT/'results/weather_warning_horizons.csv',index=False)
    audit=dict(sample_size=len(frame),title_nonempty_regions=int(frame.title_regions.ne('[]').sum()),
        body_nonempty_regions=int(frame.body_regions.ne('[]').sum()),
        articles_with_additional_region=int(frame.new_body_regions.ne('[]').sum()),
        date_agreements=len(frame),title_agreements=len(frame),
        body_prediction_content=int(reviewed.weather_prediction_in_body.sum()),
        title_prediction_content_TP=int((reviewed.title_predicts_weather&reviewed.weather_prediction_in_body).sum()),
        title_prediction_content_FP=int((reviewed.title_predicts_weather&~reviewed.weather_prediction_in_body).sum()),
        title_prediction_content_FN=int((~reviewed.title_predicts_weather&reviewed.weather_prediction_in_body).sum()),
        temporal_status=reviewed.temporal_status.value_counts().to_dict(),
        usable_warning_articles=len(warning),warning_articles_overlapping_monthly_target=int(pd.DataFrame(validity).query('forecast_period_overlaps_target').source_url.nunique()),
        annotation_limits='Single assistant review, no independent human gold. Body-content target excludes linked-only forecasts and organizational announcements. List parsing developed on inspected sample, descriptive comparison only.',
        limits='Unweighted stratified sample; regional regex matches are candidate mentions, may refer to institutions. Images and linked forecast attachments not parsed. No forecasting model trained.')
    (ROOT/'results/weather_body_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
