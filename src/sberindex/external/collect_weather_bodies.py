"""Fetch only the frozen stratified sample; preserve failures and source hashes."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
import pandas as pd

from sberindex.paths import ROOT
OUT=ROOT/'data/external/weather_body_sample'


def fetch(row):
    article_id=row['source_url'].rstrip('/').split('/')[-1]
    path=OUT/f'{article_id}.html'
    record={'source_url':row['source_url'],'path':str(path.relative_to(ROOT))}
    if not path.exists():
        temp=path.with_suffix('.partial')
        result=subprocess.run(['curl','-LsS','--fail','--max-time','30','--retry','1',row['source_url'],'-o',str(temp)],capture_output=True,text=True)
        if result.returncode:
            return dict(record,status='failed',error=result.stderr.strip())
        temp.replace(path)
    return dict(record,status='downloaded',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    sample=pd.read_csv(OUT/'sample.csv')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fetch,sample.to_dict('records')))
    manifest={'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sample_sha256':hashlib.sha256((OUT/'sample.csv').read_bytes()).hexdigest(),
        'tls':'Default curl certificate verification enabled','pages':rows}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(pd.Series([r['status'] for r in rows]).value_counts().to_dict())
    if any(r['status']!='downloaded' for r in rows):raise RuntimeError('Incomplete collection; rerun to fetch missing pages')


if __name__=='__main__':main()
