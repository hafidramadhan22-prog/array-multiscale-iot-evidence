"""Render all revised documents, validate glyphs/refs/bounds, create review sheets."""
from pathlib import Path
import json,re,hashlib,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'paper/array_20261105_editorial';OUT=ROOT/'tmp/pdfs/array_editorial_20261105';OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for name,rel in [('main','elsarticle/main'),('supplement','elsarticle/supplement')]:
    src=BASE/rel;log=src.with_suffix('.log').read_text();assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I),name
    d=fitz.open(src.with_suffix('.pdf'));text=unicodedata.normalize('NFKC','\n'.join(p.get_text() for p in d));assert '[?]' not in text and '\ufffd' not in text
    if name=='main':
        assert 'AGERS' not in text and 'author approval pending' not in text and 'all authors is pending' not in text
        assert (BASE/'elsarticle/main.bbl').read_text().count(r'\bibitem')==25
        for token in ['42.63','42.24','41.75','1.1425','0.3203','hafid1050677@gmail.com']:assert token in text,token
    tex=src.with_suffix('.tex').read_text();pages=[]
    for i,p in enumerate(d):
        assert len(p.get_text().strip())>20
        for b in p.get_text('blocks'):assert b[0]>=-.5 and b[1]>=-.5 and b[2]<=p.rect.width+.5 and b[3]<=p.rect.height+.5,(name,i+1,b[:4])
        path=OUT/f'{name}_{i+1:02d}.png';p.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(path);pages.append(path)
    sheets=[]
    for start in range(0,len(pages),4):
        ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]];w,h=ims[0].size;sheet=Image.new('RGB',(w*2+24,(h+28)*2+24),'#dce3eb');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(ims):
            x=12+(j%2)*w;y=12+(j//2)*(h+28);draw.text((x+8,y+5),f'{name.upper()} PAGE {start+j+1}',fill='black');sheet.paste(im,(x,y+28))
        path=OUT/f'{name}_sheet_{start//4+1:02d}.png';sheet.save(path);sheets.append(path.relative_to(ROOT).as_posix())
    (OUT/(name+'_text.txt')).write_text(text,encoding='utf8')
    rows.append(dict(document=name,pages=len(d),tables=tex.count(r'\begin{table}')+(1 if name=='response' else 0),figures=tex.count(r'\begin{figure}'),sha256=hashlib.sha256(src.with_suffix('.pdf').read_bytes()).hexdigest(),visual_review='pending',clipping=[],sheets=sheets))
(ROOT/'reports/ARRAY_EDITORIAL_20261105/pdf_quality.json').write_text(json.dumps(rows,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='sheets'} for r in rows],indent=2))
