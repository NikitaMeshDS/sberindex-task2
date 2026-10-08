from pathlib import Path
from bs4 import BeautifulSoup
import requests,json,datetime,hashlib,concurrent.futures,re,urllib3
from urllib.parse import urljoin
urllib3.disable_warnings()
b=Path(__file__).resolve().parent
s=BeautifulSoup((b/'bashkortostan_labor.html').read_bytes(),'html.parser')
node=s.find('a',string=lambda x:x and '2024г' in x and 'заработная' in x).parent.parent
items=[]
for n in node.select('.document-list__item--row'):
 title=n.select_one('.document-list__item-title').get_text(' ',strip=True)
 info=n.select_one('.document-list__item-info').get_text(' ',strip=True)
 date=info.split(', ')[-1]; release=datetime.datetime.strptime(date,'%d.%m.%Y').date()
 items.append({'title':title,'published_at':str(release),'available_from':str(release+datetime.timedelta(days=1)),'url':urljoin('https://02.rosstat.gov.ru',n.select_one('a[href]')['href'])})
def get(item):
 name=item['url'].split('/')[-1]; filename='bash_wage_'+str(13-items.index(item)).zfill(2)+'.pdf'
 item.update(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tls_verification=False,publication_evidence='bashkortostan_labor.html')
 try:
  r=requests.get(item['url'],timeout=30,verify=False); item.update(http_status=r.status_code,content_type=r.headers.get('Content-Type'))
  if r.ok and r.content.startswith(b'%PDF'):
   (b/filename).write_bytes(r.content); item.update(path=filename,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content),status='captured')
  else:item.update(status='http_or_content_failure')
 except Exception as e:item.update(status='request_failure',error=str(e))
 return item
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(get,items))
(b/'wage_release_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print([(r.get('path'),r['status']) for r in rows])
