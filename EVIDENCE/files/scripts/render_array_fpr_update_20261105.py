"""Render every final page and inventory new tables, bounds and old-page identity."""
from pathlib import Path
import json,re,hashlib,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'paper/array_20261105_fpr_update/elsarticle';OLD=ROOT/'paper/array_20261105_editorial/elsarticle';OUT=ROOT/'tmp/pdfs/array_fpr_update_20261105';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for name in ['main','supplement']:
    p=B/(name+'.pdf');doc=fitz.open(p);old=fitz.open(OLD/(name+'.pdf'));pages=[];identity=[];newtables=[];clipping=[]
    log=(B/(name+'.log')).read_text(encoding='utf8');assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I)
    text=unicodedata.normalize('NFKC','\n'.join(page.get_text() for page in doc));assert '[?]' not in text and '\ufffd' not in text
    for i,page in enumerate(doc):
        assert len(page.get_text().strip())>20
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4]
            if min(x0,y0)<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:clipping.append([i+1,list(block[:4])])
        pix=page.get_pixmap(matrix=fitz.Matrix(1.15,1.15),alpha=False);path=OUT/f'{name}_{i+1:02d}.png';pix.save(path);pages.append(path)
        if i<len(old):
            op=old[i].get_pixmap(matrix=fitz.Matrix(1.15,1.15),alpha=False)
            if op.samples==pix.samples:identity.append(i+1)
        tt=unicodedata.normalize('NFKC',page.get_text())
        nums=re.findall(r'Table (S?\d+)\s*:',tt)
        relevant=[n for n in nums if (n.startswith('S') and int(n[1:])>=46) or (not n.startswith('S') and int(n)>=9)]
        if relevant:newtables.append(dict(page=i+1,tables=relevant))
    assert not clipping,(name,clipping)
    for token in (['0/760','97.8800','61.109','42.24','41.75'] if name=='main' else ['Table S46:','Table S65:','Table S1:','Table S45:']):assert token in text,(name,token)
    (OUT/(name+'_text.txt')).write_text(text,encoding='utf8')
    sheets=[]
    for start in range(0,len(pages),4):
        ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]];w,h=ims[0].size;sheet=Image.new('RGB',(2*w+24,2*(h+24)+24),'#dfe7ed');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(ims):
            x=12+(j%2)*w;y=12+(j//2)*(h+24);draw.text((x+6,y+4),f'{name.upper()} {start+j+1}',fill='black');sheet.paste(im,(x,y+24))
        path=OUT/f'{name}_sheet_{start//4+1:02d}.png';sheet.save(path);sheets.append(path.relative_to(ROOT).as_posix())
    records.append(dict(document=name,pages=len(doc),sha256=sha(p),tables=(11 if name=='main' else 65),figures=(B/(name+'.tex')).read_text(encoding='utf8').count(r'\begin{figure}'),clipping=clipping,pixel_identical_previous_pages=identity,new_table_pages=newtables,sheets=sheets,visual_review='PENDING'))
(ROOT/'reports/ARRAY_FPR_UPDATE_20261105/pdf_quality.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
print(json.dumps([{k:v for k,v in r.items() if k!='sheets'} for r in records],indent=2))
