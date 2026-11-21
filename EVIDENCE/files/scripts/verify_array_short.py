"""Check the page cap, source traceability and render both revised PDFs."""
from pathlib import Path
import hashlib,json,re,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'paper/array_20261022_20pages';SRC=BASE/'elsarticle'
OUT=ROOT/'tmp/pdfs/array_short_20261022';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    evidence=json.loads((BASE/'evidence/evidence.json').read_text())
    for rec in evidence['source_records']:assert sha(ROOT/rec['path'])==rec['sha256'],rec['path']
    inv=json.loads((BASE/'template_inventory.json').read_text())
    for rec in inv['files']:assert sha(BASE/rec['path'])==rec['sha256'],rec['path']
    assert sha(ROOT/'output/pdf/ARRAY_M_HAFID_RAMADHAN_SATRIA_MANDALA.pdf')=='62544ac3065e1d33c3605cb6253b1ccda8af3be3ca1d43b2de8751dcad1e316b'
    records=[]
    for stem in ['main','supplement']:
        source=(SRC/(stem+'.tex')).read_text(encoding='utf8');log=(SRC/(stem+'.log')).read_text(encoding='utf8')
        assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I),stem
        assert not '{{' in source and r'preprint,12pt' in source
        assert not any(x in source for x in ['geometry','newgeometry','textheight','textwidth=','\\tiny','\\scriptsize'])
        doc=fitz.open(SRC/(stem+'.pdf'));text=unicodedata.normalize('NFKC','\n'.join(p.get_text() for p in doc))
        if stem=='main':
            assert len(doc)<=20
            assert (SRC/'main.bbl').read_text().count(r'\bibitem')==25
            for word in ['M Hafid Ramadhan','Satria Mandala','97.8954','34.46','0.3297','0.3310','91.0320','13,171','52,684','Declarations']:
                assert word in text,word
            assert len(re.findall(r'\\begin\{table\}',source))==4
            assert len(re.findall(r'\\begin\{figure\}',source))==3
        else:
            for word in ['S1:','S12:','Historical holdout','Resampling']:
                assert word in text,word
        pages=[];bad=[]
        for i,page in enumerate(doc):
            assert len(page.get_text().strip())>80,(stem,i+1)
            for b in page.get_text('blocks'):
                if b[0]<-.5 or b[1]<-.5 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5:bad.append((i+1,b[:4]))
            path=OUT/f'{stem}_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(path);pages.append(path)
        assert not bad,bad
        sheets=[]
        for start in range(0,len(pages),4):
            ims=[Image.open(p).convert('RGB') for p in pages[start:start+4]];w=max(x.width for x in ims);h=max(x.height for x in ims)
            sheet=Image.new('RGB',(2*w+24,2*(h+30)+24),'#dce3eb');draw=ImageDraw.Draw(sheet)
            for j,im in enumerate(ims):
                x=12+(j%2)*w;y=12+(j//2)*(h+30);draw.text((x+12,y+5),f'{stem.upper()} PAGE {start+j+1}',fill='black');sheet.paste(im,(x,y+30))
            p=OUT/f'{stem}_sheet_{start//4+1:02d}.png';sheet.save(p);sheets.append(str(p.relative_to(ROOT)).replace('\\','/'))
        (OUT/(stem+'_text.txt')).write_text(text,encoding='utf8')
        records.append(dict(name=stem,pages=len(doc),sha256=sha(SRC/(stem+'.pdf')),overfull_boxes=False,undefined_citations=False,
            out_of_page_blocks=bad,contact_sheets=sheets,rendered_pages=len(pages),page_image_sha256={p.name:sha(p) for p in pages}))
    result=dict(status='AUTOMATED_CHECKS_PASSED_VISUAL_PENDING',page_cap_main_including_references=20,
        original_font_and_margins=True,verified_source_records=len(evidence['source_records']),verified_template_files=len(inv['files']),
        old_pdf_preserved=True,scientific_results_changed=False,new_training=False,pdfs=records)
    (BASE/'quality_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'pdfs':[{k:v for k,v in r.items() if k!='page_image_sha256'} for r in records]},indent=2))
if __name__=='__main__':main()
