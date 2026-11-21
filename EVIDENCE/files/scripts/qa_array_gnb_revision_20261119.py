"""Render every final page and check fonts, references, numbering and bounds."""
from pathlib import Path
import json,re,shutil,hashlib
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'paper/array_20261119_gnb/latex'
OUT=ROOT/'reports/ARRAY_GNB_REVISION_20261119'
RENDER=ROOT/'tmp/pdfs/array_gnb_revision_20261119'
DOCS={
 'main':'ARRAY_HAFID_GNB_REVISED_20261119.pdf',
 'supplement':'ARRAY_HAFID_SUPPLEMENT_S1_S128_GNB_20261119.pdf',
 'cover_letter':'ARRAY_HAFID_COVER_LETTER_GNB_20261119.pdf',
 'revision_log':'ARRAY_HAFID_GNB_REVISION_LOG_20261119.pdf'}

def main():
    records=[]
    for name,filename in DOCS.items():
        p=ROOT/'output/pdf'/filename;shutil.copy2(SRC/(name+'.pdf'),p)
        folder=RENDER/name;folder.mkdir(parents=True,exist_ok=True)
        d=fitz.open(p);texts=[];bad=[];thumbs=[];fonts=set()
        for i,page in enumerate(d):
            text=page.get_text();assert '??' not in text and '\ufffd' not in text,(name,i)
            texts.append(text)
            for block in page.get_text('dict')['blocks']:
                if block['type']!=0:continue
                for line in block['lines']:
                    for sp in line['spans']:
                        x0,y0,x1,y1=sp['bbox']
                        if min(x0,y0)<-1 or x1>page.rect.width+1 or y1>page.rect.height+1:bad.append({'page':i+1,'text':sp['text'],'bbox':sp['bbox']})
            for f in page.get_fonts(full=True):
                if f[0]:assert len(d.extract_font(f[0])[3])>0;fonts.add(f[3])
            pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False);pix.save(str(folder/f'page_{i+1:03}.png'))
            im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((470,650))
            card=Image.new('RGB',(490,690),'#e5e7eb');card.paste(im,((490-im.width)//2,25));ImageDraw.Draw(card).text((12,6),f'{name}: PDF page {i+1}',fill='black');thumbs.append(card)
        for j in range(0,len(thumbs),9):
            sheet=Image.new('RGB',(1470,2070),'white')
            for k,im in enumerate(thumbs[j:j+9]):sheet.paste(im,((k%3)*490,(k//3)*690))
            sheet.save(folder/f'contact_{j//9+1:02}.png')
        full='\n'.join(texts);(OUT/(name+'_extracted.txt')).write_text(full,encoding='utf-8')
        captions=re.findall(r'^Table (S?\d+):',full,re.M);figures=re.findall(r'^Figure (S?\d+):',full,re.M)
        if name in ['main','supplement']:
            pref='S' if name=='supplement' else '';nt=128 if pref else 7;nf=10 if pref else 5
            assert captions==[pref+str(i) for i in range(1,nt+1)],(name,captions)
            assert figures==[pref+str(i) for i in range(1,nf+1)],(name,figures)
        if name=='main':
            assert all(x in full for x in ['Gaussian','95.05','43.92','93.77','97.90','S116','S124','S125','S126','S127','37.40','41.75'])
        assert not bad,(name,bad)
        pages={}
        for i,t in enumerate(texts):
            for hit in re.findall(r'(?:Table S\d+:|Figure S\d+:|Table \d+:|Figure \d+:)',t):pages[hit]=i+1
        record=dict(name=name,path=p.relative_to(ROOT).as_posix(),pages=len(d),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bounds=bad,embedded_fonts=sorted(fonts),tables=captions,figures=figures,caption_pages=pages,render_folder=folder.relative_to(ROOT).as_posix(),visual_review='PENDING')
        records.append(record);print(name,len(d),'pages',len(captions),'tables',len(figures),'figures',flush=True)
    (OUT/'pdf_quality.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
