"""Archive primary bibliographic records; no paywall bypass or submission."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request, urllib.error
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/array_20261021';OUT.mkdir(parents=True,exist_ok=True)
DOIS=['10.1016/j.array.2023.100306','10.1016/j.array.2025.100406',
      '10.1016/j.array.2026.101012','10.1016/j.array.2026.101114','10.1016/j.ins.2026.123908']
def fetch(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Academic manuscript preparation (local research audit)'}),timeout=40) as f:
            return f.status,f.read(),f.headers.get('Content-Type','')
    except Exception as e:return str(e),b'',''
def record(doi):
    status,body,ctype=fetch('https://api.crossref.org/works/'+doi)
    if not body:return {'doi':doi,'status':status}
    m=json.loads(body)['message'];name=doi.rsplit('/',1)[-1]
    (OUT/(name+'.json')).write_bytes(body)
    return {k:m.get(k) for k in ['DOI','title','author','container-title','volume','issue','article-number','page','published','published-online','published-print','created','link','license','relation']}
def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(record,DOIS))
    (OUT/'crossref_summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    urls={
      'comparative_array.pdf':'https://zuscholars.zu.ac.ae/cgi/viewcontent.cgi?article=8359&context=works',
      'fedshap_readme.md':'https://raw.githubusercontent.com/Deepan777/FedSHAP-WSN/main/README.md',
      'guide.html':'https://www.sciencedirect.com/journal/array/publish/guide-for-authors',
      'ensemble_array.html':'https://www.sciencedirect.com/science/article/pii/S2590005623000310',
      'adsb_array.html':'https://www.sciencedirect.com/science/article/pii/S2590005626003358'}
    access=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures={ex.submit(fetch,u):(n,u) for n,u in urls.items()}
        for future in concurrent.futures.as_completed(futures):
            n,u=futures[future];status,body,ctype=future.result()
            saved=False
            if status==200:
                (OUT/n).write_bytes(body);saved=True
            access.append(dict(url=u,status=status,content_type=ctype,saved=saved,bytes=len(body),sha256=hashlib.sha256(body).hexdigest() if saved else None))
    (OUT/'access_log.json').write_text(json.dumps(access,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'metadata':rows,'access':access},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
