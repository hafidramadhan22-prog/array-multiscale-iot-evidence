"""Exact Decimal timestamp comparison for lossless PCAPNG conversion metadata.

The inherited code compared text ('...040961' vs '...040961000'). No model
outputs had been inspected when this pre-scoring validation fix was made.
"""
from pathlib import Path
import argparse,json,sys,shutil
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_independent_capture_20261109 as r
import ta_hafid_engine as e
original_capmeta=r.capmeta;original_freeze=r.freeze
def canonical_capmeta(path):
    out=original_capmeta(path)
    for key in ['Start time','End time']:
        out['original_'+key+'_text']=out[key];out[key]=str(Decimal(out[key].replace(',','.')).normalize())
    return out
def freeze():
    original_freeze();p=Path(__file__);dest=r.RUN/'execution_source_manifest_v2.json'
    if dest.exists():
        row=r.read(dest);assert e.sha256(p)==row['sha256']
    else:
        assert not any(r.RUN.glob('*/scored.json')),'Fix must precede first external scoring'
        shutil.copy2(p,r.RUN/'source'/p.name);r.write(dest,dict(path=p.relative_to(ROOT).as_posix(),sha256=e.sha256(p),purpose='Canonical Decimal representations, exact time equality unchanged; no probability/model changes'))
        r.event('DECIMAL_TIME_REPRESENTATION_FIX_FROZEN_BEFORE_EXTERNAL_SCORING')
r.capmeta=canonical_capmeta;r.freeze=freeze
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','score','summarize'],required=True);ap.add_argument('--source',choices=list(r.SOURCES));a=ap.parse_args()
    if a.phase=='prepare':
        folder=r.RUN/a.source
        if folder.exists() and not any(folder.iterdir()):
            archive=r.RUN/'preparation_initial_failures';archive.mkdir(exist_ok=True);target=archive/(a.source+'_empty_after_text_precision_gate')
            assert folder.resolve().is_relative_to(ROOT) and target.resolve().is_relative_to(ROOT) and not target.exists();folder.rename(target)
        r.prepare(a.source)
    elif a.phase=='score':r.score(a.source)
    else:r.summarize()
