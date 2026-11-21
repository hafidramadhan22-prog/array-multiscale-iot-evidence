"""Render every final page and validate inventories before visual inspection."""
from pathlib import Path
import json,hashlib,re,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_FINAL_INTEGRATION_20261110';IMG=ROOT/'tmp/pdfs/array_final_integration_20261110';IMG.mkdir(parents=True,exist_ok=True)
FILES=[('main','ARRAY_HAFID_REVISED_GUARD_EXTERNAL_20261110.pdf',15,8),('supplement','ARRAY_HAFID_SUPPLEMENT_S1_S95_20261110.pdf',95,6),('audit','ARRAY_HAFID_AUDIT_REVISI_DOSPEM_20261110.pdf',None,None)]
rows=[]
for name,filename,nt,nf in FILES:
    path=ROOT/'output/pdf'/filename;doc=fitz.open(path);texts=[];pages=[];bounds=[];tables=[];figures=[]
    for i,pg in enumerate(doc):
        text=unicodedata.normalize('NFKC',pg.get_text());texts.append(text)
        assert len(text.strip())>40,(name,i+1,'Almost empty page')
        for b in pg.get_text('blocks'):
            if min(b[0],b[1])<-.5 or b[2]>pg.rect.width+.5 or b[3]>pg.rect.height+.5:bounds.append([i+1,list(b[:4])])
        tables.extend(dict(number=x,page=i+1) for x in re.findall(r'^Table (S?\d+)\s*:',text,re.M))
        figures.extend(dict(number=x,page=i+1) for x in re.findall(r'^Figure (S?\d+)\s*:',text,re.M))
        p=IMG/f'{name}_{i+1:02d}.png';pg.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(p);pages.append(p)
    full='\n'.join(texts);assert '[?]' not in full and '\ufffd' not in full
    if nt:
        expected=[('S' if name=='supplement' else '')+str(i) for i in range(1,nt+1)]
        assert [r['number'] for r in tables]==expected,(name,tables)
        assert len(figures)==nf,(name,figures)
    if name=='main':
        for term in ['1.0104','0.1435','31.90','18.85','1,657/2,000','41.75','0.3203','0.3270','hafid1050677@gmail.com']:
            assert term in full,(name,term)
        assert 'author approval pending' not in full.lower()
    if name=='audit':assert len(doc)==5
    sheets=[]
    for start in range(0,len(pages),4):
        ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]];w,h=ims[0].size
        sheet=Image.new('RGB',(w*2+24,(h+24)*2+24),'#dce5ec');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(ims):
            x=12+(j%2)*w;y=12+(j//2)*(h+24);draw.text((x+5,y+3),f'{name.upper()} {start+j+1}',fill='black');sheet.paste(im,(x,y+22));im.close()
        target=IMG/f'{name}_sheet_{start//4+1:02d}.png';sheet.save(target);sheets.append(target.relative_to(ROOT).as_posix())
    (IMG/(name+'_text.txt')).write_text(full,encoding='utf8')
    rows.append(dict(document=name,path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,pages=len(doc),tables=tables,figures=figures,bounds=bounds,sheets=sheets,visual_review='PENDING'))
    print(name,len(doc),'pages; tables',len(tables),'figures',len(figures),'bounds',bounds,flush=True)
(OUT/'pdf_quality.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
(OUT/'pdf_outputs.json').write_text(json.dumps(dict(pdfs=[{k:v for k,v in r.items() if k in ['document','path','sha256','bytes','pages']} for r in rows]),indent=2)+'\n',encoding='utf8')
