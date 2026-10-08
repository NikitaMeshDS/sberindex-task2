from pathlib import Path
import json,csv,re,hashlib
from datetime import date,timedelta
from bs4 import BeautifulSoup
b=Path(__file__).resolve().parent
releases=json.loads((b/'wage_release_manifest.json').read_text()); rows=[]
for i,r in enumerate(releases):
 m=12-i
 if r['status']!='captured':continue
 t=(b/r['path']).with_suffix('.txt').read_text();v,y=re.search(r'Всего\s+([\d,]+)\s+([\d,]+)',t).groups()
 rows.append({'region_name':'Республика Башкортостан','region_code':'02','period_start':'2024-01','period_end':f'2024-{m:02}','series_kind':'cumulative_period_mean_nominal_wage','nominal_rub':float(v.replace(',','.')),'yoy_pct':float(y.replace(',','.')),'published_at':r['published_at'],'available_from':r['available_from'],'source_url':r['url'],'source_path':r['path'],'release_evidence':'bashkortostan_labor.html','asof_training_2023_eligible':False})
with (b/'bashkortostan_wage_releases_2024.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
warnings=[]
for source in ['weather_body_sample/warning_events.csv','regional_news_expansion/event_registry.csv']:
 p=b.parent/source
 rs=list(csv.DictReader(p.open()))
 out=[]
 for r in rs:
  pub=r.get('published_date','');start=r.get('event_start','');av=r.get('available_from','')
  out.append({'source_url':r['source_url'],'published_at':pub,'event_start':start,'available_from':av,'published_strictly_before_start':bool(pub and start and pub<start),'available_strictly_before_start':bool(av and start and av<start),'municipal_id':r.get('territory_id'),'limits':r.get('scope_limits',r.get('scope'))})
 warnings.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'row_count':len(rs),'published_before_start_count':sum(x['published_strictly_before_start'] for x in out),'available_before_start_count':sum(x['available_strictly_before_start'] for x in out),'rows':out})
(b/'advance_warning_audit.json').write_text(json.dumps(warnings,ensure_ascii=False,indent=2))
s=BeautifulSoup((b/'rosstat_living.html').read_bytes(),'html.parser');cand=[]
for a in s.select('a[href]'):
 if any(k in a.get('href','') for k in ['urov_10','urov_11','urov_12']):
  p=a.find_parent(class_='document-list__item'); cand.append({'url':'https://rosstat.gov.ru'+a['href'],'listing_text':p.get_text(' ',strip=True) if p else ''})
(b/'income_candidates.json').write_text(json.dumps(cand,ensure_ascii=False,indent=2))
print('wages',len(rows));print([(x['path'],x['published_before_start_count'],x['available_before_start_count']) for x in warnings])
