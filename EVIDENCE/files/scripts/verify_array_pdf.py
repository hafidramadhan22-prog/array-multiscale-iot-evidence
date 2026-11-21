"""Verify citations/evidence and render every page for a separate visual review."""
from pathlib import Path
import hashlib,json,re,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper/array_20261021';SRC=BASE/'elsarticle'
OUT=ROOT/'tmp/pdfs/array_final';OUT.mkdir(parents=True,exist_ok=True)

def main():
    evidence=json.loads((BASE/'evidence/evidence.json').read_text())
    for rec in evidence['source_records']:
        p=ROOT/rec['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],p
    inv=json.loads((BASE/'template_inventory.json').read_text())
    for rec in inv['files']:
        p=BASE/rec['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],p
    tex=(SRC/'main.tex').read_text(encoding='utf8');log=(SRC/'main.log').read_text(encoding='utf8')
    assert r'\\' not in tex[:50] # normal source, not JSON-escaped newlines
    assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I),log[-500:]
    assert not re.search(r'\{\{|r0[,}]',tex)
    assert 'elsarticle.dtx' in (SRC/'elsarticle.cls').read_text(encoding='utf8')
    bib=(SRC/'main.bbl').read_text(encoding='utf8');assert bib.count(r'\bibitem')==25
    doc=fitz.open(SRC/'main.pdf');text=unicodedata.normalize('NFKC','\n'.join(p.get_text() for p in doc))
    assert len(doc)>20 and all(len(p.get_text().strip())>100 for p in doc)
    for term in ['M Hafid Ramadhan','Satria Mandala','satriamandala@telkomuniversity.ac.id','Telkom University','97.895','0.6626','0.3297','Local prototype']:
        assert term in text,term
    assert not any(s in text for s in ['{{TABLE','{{FIGURE','[?]','Preprint submitted'])
    bad=[];page_info=[];pages=[]
    for i,page in enumerate(doc):
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4]
            if x0<-0.5 or y0<-0.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:bad.append((i+1,block[:4]))
        png=OUT/f'page_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(png)
        pages.append(png);page_info.append(dict(page=i+1,words=len(page.get_text('words')),images=len(page.get_images()),dimensions=list(page.rect)))
    assert not bad,bad
    sheets=[]
    for start in range(0,len(pages),4):
        ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]]
        width=max(im.width for im in ims);height=max(im.height for im in ims)
        sheet=Image.new('RGB',(2*width+24,2*(height+30)+24),'#dce3eb');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(ims):
            x=12+(j%2)*width;y=12+(j//2)*(height+30)
            draw.text((x+12,y+5),f'PAGE {start+j+1}',fill='black');sheet.paste(im,(x,y+30))
        sp=OUT/f'sheet_{start//4+1:02d}.png';sheet.save(sp);sheets.append(str(sp.relative_to(ROOT)))
    result=dict(status='AUTOMATED_PDF_CHECKS_PASSED_VISUAL_PENDING',pages=len(doc),references=25,tables=15,figures=4,
        verified_source_records=len(evidence['source_records']),verified_template_files=len(inv['files']),
        clipping_blocks=bad,undefined_citations=False,overfull_boxes=False,rendered_pages=len(pages),
        contact_sheets=sheets,page_details=page_info,pdf_sha256=hashlib.sha256((SRC/'main.pdf').read_bytes()).hexdigest())
    (BASE/'pdf_quality_check.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'extracted_text.txt').write_text(text,encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='page_details'},indent=2))
if __name__=='__main__':main()
