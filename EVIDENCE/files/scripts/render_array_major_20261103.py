"""Render every final PDF page and check references/layout before visual review."""
from pathlib import Path
import re,json,hashlib,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'paper/array_20261103_major_revision/elsarticle';OUT=ROOT/'tmp/pdfs/array_major_20261103';OUT.mkdir(parents=True,exist_ok=True)
reports=[]
for name in ['main','supplement']:
    log=(SRC/(name+'.log')).read_text(encoding='utf8');assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I)
    tex=(SRC/(name+'.tex')).read_text(encoding='utf8');doc=fitz.open(SRC/(name+'.pdf'));text=unicodedata.normalize('NFKC','\n'.join(p.get_text() for p in doc))
    assert '[?]' not in text and '\ufffd' not in text
    if name=='main':
        assert (SRC/'main.bbl').read_text().count(r'\bibitem')==25
        for t in ['hafid1050677@gmail.com','Satria Mandala','42.63','41.75','97.8954','93.7694']:assert t in text,t
    pages=[];bounds=[]
    for i,p in enumerate(doc):
        assert len(p.get_text().strip())>20
        for b in p.get_text('blocks'):
            if b[0]<0 or b[1]<0 or b[2]>p.rect.width+.5 or b[3]>p.rect.height+.5:bounds.append([i+1,list(b[:4])])
        path=OUT/f'{name}_{i+1:02d}.png';p.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(path);pages.append(path)
    assert not bounds,bounds
    sheets=[]
    for start in range(0,len(pages),4):
        ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]];w=ims[0].width;h=ims[0].height
        sheet=Image.new('RGB',(w*2+24,(h+28)*2+24),'#dce3eb');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(ims):
            x=12+(j%2)*w;y=12+(j//2)*(h+28);draw.text((x+8,y+5),f'{name.upper()} PAGE {start+j+1}',fill='black');sheet.paste(im,(x,y+28))
        path=OUT/f'{name}_sheet_{start//4+1:02d}.png';sheet.save(path);sheets.append(str(path.relative_to(ROOT)))
    (OUT/(name+'_text.txt')).write_text(text,encoding='utf8')
    reports.append(dict(document=name,pages=len(doc),figures=tex.count(r'\begin{figure}'),tables=tex.count(r'\begin{table}'),sha256=hashlib.sha256((SRC/(name+'.pdf')).read_bytes()).hexdigest(),clipping=[],sheets=sheets,visual_review='pending'))
(ROOT/'reports/ARRAY_MAJOR_REVISION_20261103/pdf_quality.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
