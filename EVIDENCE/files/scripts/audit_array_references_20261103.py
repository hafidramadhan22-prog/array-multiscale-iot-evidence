"""Archive publisher-deposited DOI metadata; distinguish access from verification."""
from pathlib import Path
import re,json,urllib.request,concurrent.futures,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/array_20261103';OUT.mkdir(exist_ok=True)
source=(ROOT/'paper/array_20261022_integrated/elsarticle/references.bib').read_text()
entries=[]
for block in source.split('@misc{')[1:]:
    key=block.split(',')[0];doi=re.search(r'doi = \{([^}]+)',block);url=re.search(r'url = \{([^}]+)',block)
    entries.append(dict(key=key,doi=doi[1] if doi else None,url='https://api.crossref.org/works/'+doi[1] if doi else url[1]))
def fetch(row):
    try:
        with urllib.request.urlopen(urllib.request.Request(row['url'],headers={'User-Agent':'Research bibliography verification'}),timeout=40) as r:body=r.read();status=r.status
        suffix='.json' if row['doi'] else '.html';(OUT/(row['key']+suffix)).write_bytes(body)
        row.update(status=status,sha256=hashlib.sha256(body).hexdigest(),file=row['key']+suffix)
        if row['doi']:row['metadata']=json.loads(body)['message']
    except Exception as ex:row.update(status='unavailable',error=str(ex))
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(fetch,entries))
(OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps([dict(key=r['key'],status=r['status'],title=r.get('metadata',{}).get('title')) for r in result],indent=2))
