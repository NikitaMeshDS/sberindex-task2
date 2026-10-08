"""Offline completeness and conservative geography audit of official news titles.

Regional mention counts are source-observation features, NOT hazard labels.
No forecasting claim is made without reading article bodies and auditing scope.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import pandas as pd

from sberindex.paths import ROOT
MONTHS={name:i+1 for i,name in enumerate('января февраля марта апреля мая июня июля августа сентября октября ноября декабря'.split())}


def title_type(title):
    text=title.lower().replace('ё','е')
    if re.search(r'предупреждени|штормов',text):return 'warning_title'
    if re.search(r'прогноз',text):return 'forecast_title'
    if re.search(r'обзор погод|обстановк|по состоянию|обзор.*климат',text):return 'observation_or_review_title'
    return 'unclassified_title'


def parse_index(text):
    # Limit parsing to the news list: navigation and sidebar links are excluded.
    text=text.split('<a name="list"></a>',1)[1].split('<a class="prev',1)[0]
    pattern=r'<strong>(\d{1,2})\s+([а-я]+)\s+(20\d{2})</strong>|<li><a href="(/press/news/\d+/)">(.*?)</a></li>'
    rows=[];date=None
    for match in re.finditer(pattern,text,re.S):
        day,month,year,url,title=match.groups()
        if day:date=f'{year}-{MONTHS[month]:02d}-{int(day):02d}'
        else:
            assert date is not None
            clean=html.unescape(re.sub('<[^>]+>','',title))
            rows.append(dict(published_date=date,source_url='https://www.meteorf.gov.ru'+url,
                             title=' '.join(clean.split())))
    return rows


def region_pattern(name):
    name=name.lower().replace('ё','е')
    if name.endswith('область'):
        adjective=name.split()[0][:-2]
        middle=r'\s+автономн\w+' if 'автономная' in name else ''
        return rf'\b{re.escape(adjective)}\w*{middle}\s+област\w*\b'
    if name.endswith('край'):
        return rf'\b{re.escape(name.split()[0][:-2])}\w*\s+кра[йяею]\b'
    if name=='республика алтай':return r'\bреспублик\w*\s+алтай\b'
    if name=='республика коми':return r'\bкоми\b'
    if name.startswith('республика '):
        term=name.split()[1]
        if term=='саха':return r'\b(?:якут\w*|республик\w*\s+саха)\b'
        if term=='марий':return r'\bмарий\s+эл\b'
        if term=='северная':return r'\bсеверн\w*\s+осети\w*\b'
        if term in {'адыгея','бурятия','ингушетия','калмыкия','карелия','мордовия','тыва','хакасия'}:term=term[:-1]
        return rf'\b{re.escape(term)}\w*\b'
    if 'республика' in name:return rf'\b{re.escape(name.split()[0][:-2])}\w*\s+республик\w*\b'
    if name=='москва':return r'\bмоскв[аеуы]\b'
    if name=='санкт-петербург':return r'\bсанкт-петербург\w*\b'
    if name=='севастополь':return r'\bсевастопол\w*\b'
    if 'автономный округ' in name:return rf'(?<![\w-]){re.escape(name.split()[0][:-2])}\w*\s+автономн\w*\s+округ\w*\b'
    raise ValueError(name)


def main():
    manifest=json.loads((ROOT/'data/external/weather_news_indexes/manifest.json').read_text())
    rows=[];counts=[]
    for page in manifest['pages']:
        path=ROOT/page['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==page['sha256'],page['path']
        parsed=parse_index(path.read_text())
        assert all(r['published_date'].startswith(str(page['year'])) for r in parsed)
        rows.extend(parsed);counts.append(dict(year=page['year'],page=page['page'],parsed=len(parsed),reported=page['reported_count']))
    articles=pd.DataFrame(rows)
    assert not articles.source_url.duplicated().any(), 'Duplicated index entries; completeness cannot be asserted'
    for year,group in pd.DataFrame(counts).groupby('year'):
        assert group.reported.nunique()==1 and group.parsed.sum()==group.reported.iloc[0],str(year)
    articles=articles.sort_values(['published_date','source_url']).reset_index(drop=True)
    articles['available_from']=(pd.to_datetime(articles.published_date)+pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    articles['title_type']=articles.title.map(title_type)
    crosswalk=pd.read_csv(ROOT/'results/region_crosswalk.csv')
    patterns={int(r.region_code):region_pattern(r.region_name) for r in crosswalk.itertuples()}
    mentions=[]
    for article in articles.itertuples():
        title=article.title.lower().replace('ё','е')
        for code,pattern in patterns.items():
            if re.search(pattern,title):
                mentions.append(dict(source_url=article.source_url,region_code=code,published_date=article.published_date,
                    available_from=article.available_from,title=article.title,title_type=article.title_type,scope='region_mentioned_in_title_only'))
    mentions=pd.DataFrame(mentions)
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    raw=pd.read_parquet(ROOT/'data/consumption.parquet',columns=['territory_id','date'])
    eligible=set(lookup[(lookup.year==2024)&lookup.territory_id.isin(raw.territory_id)].region_code)
    features=[]
    for origin in pd.date_range('2023-01-01','2024-12-01',freq='MS'):
        end=origin+pd.offsets.MonthEnd(0);start=end-pd.Timedelta(days=90)
        known=mentions[(pd.to_datetime(mentions.available_from)<=end)&(pd.to_datetime(mentions.available_from)>start)]
        n=known.groupby('region_code').source_url.nunique()
        for code in sorted(eligible):
            row=dict(origin=origin.strftime('%Y-%m'),region_code=code,explicit_title_mentions_90d=int(n.get(code,0)),
                window_left_truncated=start<pd.Timestamp('2023-01-01'))
            for kind in ['warning_title','forecast_title','observation_or_review_title','unclassified_title']:
                row[kind+'_90d']=int(known[(known.region_code==code)&(known.title_type==kind)].source_url.nunique())
            features.append(row)
    out=ROOT/'results'
    articles.to_csv(out/'weather_news_articles.csv',index=False)
    mentions.to_csv(out/'weather_news_region_mentions.csv',index=False)
    pd.DataFrame(features).to_csv(out/'weather_news_asof_features.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'weather_news_pages.csv',index=False)
    articles.groupby(['title_type',articles.published_date.str[:7]],as_index=False).size().to_csv(out/'weather_news_topic_monthly.csv',index=False)
    coverage=mentions.groupby('region_code',as_index=False).agg(articles=('source_url','nunique'))
    coverage=crosswalk[['region_code','region_name']].merge(coverage,how='left').fillna({'articles':0})
    coverage['in_sber_panel']=coverage.region_code.isin(eligible)
    coverage.to_csv(out/'weather_news_coverage.csv',index=False)
    audit=dict(articles=len(articles),by_year=articles.groupby(articles.published_date.str[:4]).size().to_dict(),
        pages=len(counts),articles_with_explicit_region=int(mentions.source_url.nunique()),
        regions_with_mentions=int(mentions.region_code.nunique()),sber_regions_with_mentions=len(set(mentions.region_code)&eligible),
        sber_regions=len(eligible),patterns=patterns,
        title_types=articles.title_type.value_counts().to_dict(),
        title_type_limits='Lexical title flags only, no body validation. Warning takes precedence over forecast and observation. Unclassified titles retained, not assumed irrelevant.',
        interpretation='Zero = no explicit mention matched in archived title, NOT no hazard. Broad geographies and bodies are not inferred. Regional references are not municipality exposure labels.',
        availability='Publication date +1 day is an assumption; archive fetched later, historical modifications unknown.',
        model_status='Not used to train or select a forecasting model; geography and article-body review remain necessary.')
    (out/'weather_news_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='patterns'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
