"""Compare clean-built PDF text/pixels and the new native figure."""
from pathlib import Path
import sys,json
import pymupdf as fitz
root=Path(sys.argv[1]);records=[]
for built,expected in [('main.pdf','01_MAIN_MANUSCRIPT_ARRAY_GNB_20261119.pdf'),('SUPPLEMENTARY/supplement.pdf','02_SUPPLEMENT_S1_S128_ARRAY_GNB_20261119.pdf'),('COVER_LETTER/cover_letter.pdf','03_COVER_LETTER_ARRAY_GNB_20261119.pdf'),('REVISION_LOG/revision_log.pdf','04_REVISION_LOG_ARRAY_GNB_20261119.pdf'),('rebuilt_gnb/gnb_private_confusion.pdf','figures/gnb_private_confusion.pdf')]:
    a=fitz.open(root/built);b=fitz.open(root/expected);assert len(a)==len(b)
    for i in range(len(a)):
        assert a[i].get_text()==b[i].get_text(),(built,i+1,'text')
        x=a[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False);y=b[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        assert x.width==y.width and x.height==y.height and x.samples==y.samples,(built,i+1,'pixels')
    records.append(dict(file=built,pages=len(a),text_identical=True,pixels_identical=True))
Path(sys.argv[2]).write_text(json.dumps(dict(status='PASS_FOUR_CLEAN_PDFS_AND_GNB_NATIVE_FIGURE',records=records),indent=2)+'\n',encoding='utf8')
print('PASS_FOUR_CLEAN_PDFS_AND_GNB_NATIVE_FIGURE')
