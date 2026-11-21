"""Render publication files and inspect captions, fonts and page bounds."""
from pathlib import Path
import pymupdf as fitz
import json,hashlib,re,shutil
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'paper/array_20261117_revision/latex'
OUT=ROOT/'reports/ARRAY_REVISION_20261117';RENDER=ROOT/'tmp/pdfs/array_20261117'
DOCS={'main':'ARRAY_HAFID_REVISED_20261117.pdf','supplement':'ARRAY_HAFID_SUPPLEMENT_S1_S104_REVISED_20261117.pdf','revision_log':'ARRAY_HAFID_REVISION_LOG_20261117.pdf','cover_letter':'ARRAY_HAFID_COVER_LETTER_20261117.pdf'}

def main():
    RENDER.mkdir(parents=True,exist_ok=True);records=[]
    for name,filename in DOCS.items():
        p=ROOT/'output/pdf'/filename;shutil.copy2(SRC/(name+'.pdf'),p)
        d=fitz.open(p);alltext='';bad=[];page_text=[];fonts=set();thumbs=[]
        folder=RENDER/name;folder.mkdir(exist_ok=True)
        for i,page in enumerate(d):
            text=page.get_text();alltext+=text+'\n';page_text.append(text)
            assert '\ufffd' not in text and '??' not in text,(name,i+1)
            for block in page.get_text('dict')['blocks']:
                if block['type']!=0:continue
                for line in block['lines']:
                    for s in line['spans']:
                        x0,y0,x1,y1=s['bbox']
                        if x0 < -1 or y0 < -1 or x1>page.rect.width+1 or y1>page.rect.height+1:bad.append(dict(page=i+1,text=s['text'],bbox=s['bbox']))
            for f in page.get_fonts(full=True):
                if f[0]:
                    extracted=d.extract_font(f[0]);assert len(extracted[3])>0,(name,i+1,f)
                    fonts.add(f[3])
            pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False);pix.save(str(folder/f'page_{i+1:03}.png'))
            im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((470,650))
            card=Image.new('RGB',(490,690),'#e5e7eb');card.paste(im,((490-im.width)//2,25));ImageDraw.Draw(card).text((12,6),f'{name}: PDF page {i+1}',fill='black');thumbs.append(card)
        assert not bad,(name,bad)
        for j in range(0,len(thumbs),9):
            sheet=Image.new('RGB',(1470,2070),'white')
            for k,im in enumerate(thumbs[j:j+9]):sheet.paste(im,((k%3)*490,(k//3)*690))
            sheet.save(folder/f'contact_{j//9+1:02}.png')
        tables=re.findall(r'^Table (S?\d+):',alltext,re.M);figures=re.findall(r'^Figure (S?\d+):',alltext,re.M)
        if name in ['main','supplement']:
            pref='S' if name=='supplement' else '';nt=104 if pref else 7;nf=9 if pref else 5
            assert tables==[pref+str(i) for i in range(1,nt+1)],(name,tables)
            assert figures==[pref+str(i) for i in range(1,nf+1)],(name,figures)
        if name=='main':assert len(d)==26 and '[30]' in alltext
        if name=='supplement':assert len(d)==93
        captions={}
        for i,t in enumerate(page_text):
            for hit in re.findall(r'(?:Table S\d+|Figure S\d+:|Table \d+|Figure \d+:)',t):captions.setdefault(hit,[]).append(i+1)
        records.append(dict(name=name,path=p.relative_to(ROOT).as_posix(),pages=len(d),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
            tables=tables,figures=figures,embedded_fonts=sorted(fonts),bounds=bad,captions= captions,render_folder=folder.relative_to(ROOT).as_posix(),visual_review='PENDING'))
        print(name,len(d),'pages; fonts',len(fonts),'table captions',len(tables),'figures',len(figures),flush=True)
    (OUT/'pdf_quality.json').write_text(json.dumps(records,indent=2)+'\n','utf8')
if __name__=='__main__':main()
