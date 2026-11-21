"""Compare clean TeX builds and rebuilt plots against the PDFs inside the ZIP."""
from pathlib import Path
import json,sys
import pymupdf as fitz
import numpy as np
root=Path(sys.argv[1]).resolve();result=[]
docs=[('main.pdf','01_MAIN_MANUSCRIPT_ARRAY_20261117.pdf',26),
      ('SUPPLEMENTARY/supplement.pdf','02_SUPPLEMENT_S1_S104_ARRAY_20261117.pdf',93),
      ('COVER_LETTER/cover_letter.pdf','03_COVER_LETTER_ARRAY_20261117.pdf',1),
      ('REVISION_LOG/revision_log.pdf','04_REVISION_LOG_ARRAY_20261117.pdf',4)]
for built,expected,pages in docs:
    a=fitz.open(root/built);b=fitz.open(root/expected);assert len(a)==len(b)==pages
    for i in range(pages):
        assert a[i].get_text()==b[i].get_text(),(built,i+1,'text')
        p=a[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False);q=b[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        assert p.width==q.width and p.height==q.height and p.samples==q.samples,(built,i+1,'render')
    result.append(dict(file=built,pages=pages,text_equal=True,pixels_equal=True))
figures=[]
for p in sorted((root/'figures').glob('*.pdf')):
    q=root/'rebuilt_figures/figures'/p.name
    a=fitz.open(p);b=fitz.open(q);assert len(a)==len(b)==1
    assert a[0].get_text()==b[0].get_text(),p.name
    x=a[0].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False);y=b[0].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
    assert x.width==y.width and x.height==y.height and x.samples==y.samples,p.name
    assert (root/'rebuilt_figures/figures'/p.with_suffix('.svg').name).is_file()
    figures.append(p.name)
assert len(figures)==14
record=dict(status='PASS_ALL_FOUR_CLEAN_BUILDS_AND_14_NATIVE_FIGURES',documents=result,figures=figures,figure_data_recomputed=False,new_fits=0,new_inference=0)
Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n','utf8');print(record['status'])
