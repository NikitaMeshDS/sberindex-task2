"""Recheck the two official URLs; verify TLS and retain responses/errors."""
from pathlib import Path
import urllib.request,urllib.error,json,hashlib,datetime,concurrent.futures
out=Path(__file__).resolve().parent
urls={'official_archive':'https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip','dataset_description':'https://sberindex.ru/api/researches/v1/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim'}
def fetch(item):
 name,url=item;record={'name':name,'url':url,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tls_verification':True}
 try:
  try:r=urllib.request.urlopen(url,timeout=35)
  except urllib.error.HTTPError as e:r=e
  with r:
   content=r.read();record.update(http_status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),last_modified=r.headers.get('Last-Modified'),etag=r.headers.get('ETag'),bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
  suffix='.zip' if name=='official_archive' and record['http_status']==200 and content[:2]==b'PK' else '.response'
  path=out/(name+suffix);path.write_bytes(content);record['snapshot']=path.name
 except (urllib.error.URLError,TimeoutError) as e:record['error']=str(e)
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(fetch,urls.items()))
(out/'acquisition.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows,ensure_ascii=False,indent=2))
