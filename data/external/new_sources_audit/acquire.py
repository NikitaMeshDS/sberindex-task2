from pathlib import Path
import requests,json,hashlib,datetime,concurrent.futures,urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
BASE=Path(__file__).resolve().parent
URLS={
'sber_description_api':'https://sberindex.ru/api/researches/v1/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim',
'sber_home':'https://sberindex.ru/',
'sber_dataset_description':'https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim',
'sber_dashboard_consumers':'https://sberindex.ru/ru/dashboards/?category=CONSUMERS',
'rosstat_labor':'https://rosstat.gov.ru/labor_market_employment_salaries',
'rosstat_retail':'https://rosstat.gov.ru/statistics/retail_trade',
'rosstat_living':'https://rosstat.gov.ru/folder/13397',
'rosstat_news_jan2024':'https://www.rosstat.gov.ru/central-news?page=51&print=1',
'bashkortostan_labor':'https://02.rosstat.gov.ru/folder/26137',
'cbr_daily':'https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2023&UniDbQuery.To=31.12.2024',
'rosstat_osn_jan2024':'https://rosstat.gov.ru/storage/mediabank/osn-01-2024.pdf',
}
def get(item):
 key,url=item; record={'id':key,'url':url,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  r=requests.get(url,timeout=35,verify=False);record['tls_verification']=False;record.update(http_status=r.status_code,final_url=r.url,content_type=r.headers.get('Content-Type'),last_modified=r.headers.get('Last-Modified'))
  suffix='.pdf' if r.content.startswith(b'%PDF') else '.html'; path=BASE/(key+suffix);path.write_bytes(r.content)
  record.update(path=str(path.relative_to(BASE)),bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),status='captured' if r.ok else 'http_failure')
 except Exception as e: record.update(status='request_failure',error=str(e))
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(get,URLS.items()))
(BASE/'acquisition_manifest_unverified_tls.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
for r in rows:print(r['id'],r['status'],r.get('http_status'),r.get('bytes'))
