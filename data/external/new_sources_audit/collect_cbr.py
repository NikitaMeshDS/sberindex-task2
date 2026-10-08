from pathlib import Path
import csv,requests,json,datetime,hashlib,concurrent.futures
from bs4 import BeautifulSoup
b=Path(__file__).resolve().parent
rows=list(csv.DictReader((b.parent/'cbr_rate_decisions_2023_2024.csv').open()))
def get(item):
 item.update(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tls_verification=True)
 try:
  r=requests.get(item['source_url'],timeout=25);p='cbr_decision_'+item['published_date']+'.html'
  (b/p).write_bytes(r.content);item.update(http_status=r.status_code,path=p,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content),status='captured' if r.ok else 'http_failure')
 except Exception as e:item.update(status='request_failure',error=str(e))
 return item
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(get,rows))
(b/'cbr_release_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
s=BeautifulSoup((b/'cbr_daily.html').read_bytes(),'html.parser')
t=[[td.get_text(' ',strip=True) for td in tr.select('td,th')] for tr in s.select('table tr')]
with (b/'cbr_daily_table.csv').open('w') as f:csv.writer(f).writerows(t)
print('captures',sum(r['status']=='captured' for r in rows),'daily_rows',len(t))
