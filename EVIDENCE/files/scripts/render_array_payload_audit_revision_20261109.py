"""Render all three deliverables and validate text, table inventory and bounds."""
from pathlib import Path
import json,hashlib,re,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reports/ARRAY_PAYLOAD_AUDIT_REVISION_20261109'
OUT=ROOT/'tmp/pdfs/array_payload_audit_revision_20261109'
OUT.mkdir(parents=True,exist_ok=True)
info=json.loads((REP/'pdf_outputs.json').read_text('utf8'))
rows=[]
for meta,name,nt,nf in zip(info['pdfs'],['main','supplement','comparison'],[12,73,None],[7,6,None]):
    p=ROOT/meta['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    d=fitz.open(p);pages=[];bounds=[];tables=[];figures=[];text=[]
    old=None
    if name!='comparison':old=fitz.open(ROOT/'paper/array_20261105_fpr_update/elsarticle'/(name+'.pdf'))
    identical=[]
    for i,page in enumerate(d):
        tt=unicodedata.normalize('NFKC',page.get_text());text.append(tt)
        assert len(tt.strip())>40,(name,i+1,'almost empty page')
        if name=='main':assert len(tt.strip())>150,(name,i+1,'orphan paragraph page')
        for b in page.get_text('blocks'):
            if min(b[0],b[1])<-.5 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5:bounds.append([i+1,list(b[:4])])
        for n in re.findall(r'^Table (S?\d+)\s*:',tt,re.M):tables.append(dict(number=n,page=i+1))
        for n in re.findall(r'^Figure (S?\d+)\s*:',tt,re.M):figures.append(dict(number=n,page=i+1))
        pix=page.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False);path=OUT/f'{name}_{i+1:02d}.png';pix.save(path);pages.append(path)
        if old and i<len(old):
            op=old[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False)
            if op.samples==pix.samples:identical.append(i+1)
    assert not bounds,(name,bounds)
    full='\n'.join(text);assert '[?]' not in full and '\ufffd' not in full
    if nt is not None:
        expected=[('S' if name=='supplement' else '')+str(i) for i in range(1,nt+1)]
        assert [x['number'] for x in tables]==expected,(name,tables)
        assert len(figures)==nf,(name,figures)
    if name=='main':
        for term in ['0.935234','0.111796','0.157671','1.1425','41.75','97.8800','0/760','822','24,301','hafid1050677@gmail.com']:
            assert term in full,(name,term)
        assert 'author approval pending' not in full.lower()
    if name=='comparison':assert 'Turnitin' in full and len(d)==5
    sheets=[]
    for start in range(0,len(pages),4):
        ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]];w,h=ims[0].size
        sheet=Image.new('RGB',(w*2+24,(h+26)*2+24),'#dce5ec');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(ims):
            x=12+(j%2)*w;y=12+(j//2)*(h+26);draw.text((x+6,y+4),f'{name.upper()} {start+j+1}',fill='black');sheet.paste(im,(x,y+24))
        target=OUT/f'{name}_sheet_{start//4+1:02d}.png';sheet.save(target);sheets.append(target.relative_to(ROOT).as_posix())
    (OUT/(name+'_text.txt')).write_text(full,encoding='utf8')
    rows.append(dict(document=name,path=meta['path'],sha256=meta['sha256'],pages=len(d),tables=tables,figures=figures,bounds=bounds,pixel_identical_previous_pages=identical,sheets=sheets,visual_review='PENDING'))
(REP/'pdf_quality.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
print(json.dumps([{k:v for k,v in r.items() if k not in ['tables','figures','sheets']} for r in rows],indent=2))
