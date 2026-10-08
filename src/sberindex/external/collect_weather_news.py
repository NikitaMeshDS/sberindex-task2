"""Download complete official annual news indexes, independent of target data.

Explicit collection command; ordinary refresh never uses the network.
"""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import re
import subprocess

from sberindex.paths import ROOT
OUT = ROOT/'data/external/weather_news_indexes'


def collect(year, page):
    url = f'https://www.meteorf.gov.ru/press/news/?YEAR={year}&PAGE_SIZE=100&PAGEN_1={page}'
    path = OUT/f'{year}_{page:02d}.html'
    if not path.exists():
        temporary = path.with_suffix('.partial')
        subprocess.run(['curl','-LsS','--fail','--max-time','45','--retry','1',url,'-o',str(temporary)],check=True)
        temporary.replace(path)
    text = path.read_text()
    pages = max(map(int,re.findall(r'data-page="(\d+)"',text)))
    count = re.search(r"\.count'\)\.html\('(\d+) новост",text)
    assert count, f'Missing annual count: {url}'
    return dict(year=year,page=page,total_pages=pages,reported_count=int(count[1]),url=url,
        path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    first=[collect(year,1) for year in [2023,2024]]
    jobs=[(x['year'],page) for x in first for page in range(2,x['total_pages']+1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=first+list(pool.map(lambda args:collect(*args),jobs))
    rows.sort(key=lambda r:(r['year'],r['page']))
    manifest=dict(retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source='Росгидромет, Пресс-центр, Новости и события',
        selection='All index pages for YEAR=2023 and YEAR=2024, PAGE_SIZE=100; no target-dependent selection',
        limits='Current archive, not historical page versions. Publication dates are website assertions; revisions unknown. Titles and URLs only, no article bodies.',
        tls='curl default certificate verification enabled',pages=rows)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({str(x['year']):x['reported_count'] for x in first}))


if __name__=='__main__':main()
