"""Render and inspect PDF inventories for the concise journal release."""
from pathlib import Path
import json,re,hashlib,unicodedata
import pymupdf as fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/ARRAY_CONCISE_20261110"
IMG=ROOT/"tmp/pdfs/array_concise_20261110"
FILES=[("main","ARRAY_HAFID_CONCISE_REVIEW_20261110.pdf",7,5),
       ("supplement","ARRAY_HAFID_SUPPLEMENT_S1_S104_CONCISE_20261110.pdf",104,9),
       ("changes","ARRAY_HAFID_CATATAN_PEMADATAN_20261110.pdf",None,None)]

def main():
    IMG.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,fn,nt,nf in FILES:
        path=ROOT/"output/pdf"/fn
        if not path.exists():
            if name=="changes":continue
            raise FileNotFoundError(path)
        doc=fitz.open(path);pages=[];texts=[];bounds=[];tables=[];figs=[]
        for i,pg in enumerate(doc):
            text=unicodedata.normalize("NFKC",pg.get_text())
            assert len(text.strip())>30,(name,i+1,"blank")
            texts.append(text)
            for b in pg.get_text("blocks"):
                if min(b[0],b[1])<-.5 or b[2]>pg.rect.width+.5 or b[3]>pg.rect.height+.5:
                    bounds.append([i+1,list(b[:4])])
            tables.extend(dict(number=x,page=i+1) for x in re.findall(r"^Table (S?\d+)\s*:",text,re.M))
            figs.extend(dict(number=x,page=i+1) for x in re.findall(r"^Figure (S?\d+)\s*:",text,re.M))
            p=IMG/f"{name}_{i+1:03d}.png"
            pg.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(p)
            pages.append(p)
        full="\n".join(texts)
        assert "[?]" not in full and "\ufffd" not in full,(name,"bad characters/refs")
        assert not bounds,(name,bounds)
        if nt:
            expected=[("S" if name=="supplement" else "")+str(i) for i in range(1,nt+1)]
            assert [t["number"] for t in tables]==expected,(name,tables)
            assert len(figs)==nf,(name,figs)
        if name=="main":
            for term in ["1.0104","0.1435","31.90","18.85","1,657/2,000","41.75","hafid1050677@gmail.com"]:
                assert term in full,(name,term)
            assert "author approval pending" not in full.lower()
            assert len(doc)<=30
        sheets=[]
        for start in range(0,len(pages),4):
            ims=[Image.open(p).convert("RGB") for p in pages[start:start+4]]
            w,h=ims[0].size
            sheet=Image.new("RGB",(2*w+24,2*(h+24)+24),"#dce5ec")
            draw=ImageDraw.Draw(sheet)
            for j,im in enumerate(ims):
                x=12+(j%2)*w;y=12+(j//2)*(h+24)
                draw.text((x+5,y+3),f"{name.upper()} {start+j+1}",fill="black")
                sheet.paste(im,(x,y+22));im.close()
            target=IMG/f"{name}_sheet_{start//4+1:02d}.png"
            sheet.save(target);sheets.append(target.relative_to(ROOT).as_posix())
        (IMG/(name+"_text.txt")).write_text(full,encoding="utf8")
        records.append(dict(document=name,path=path.relative_to(ROOT).as_posix(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                            pages=len(doc),tables=tables,figures=figs,bounds=bounds,sheets=sheets,
                            visual_review="PENDING"))
        print(name,len(doc),"pages;",len(tables),"tables;",len(figs),"figures")
    (OUT/"pdf_quality.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf8")

if __name__=="__main__":
    main()
