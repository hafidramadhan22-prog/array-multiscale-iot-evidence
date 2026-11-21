"""Read-only editorial artifact comparison; no model loading or experiments."""
from pathlib import Path
import re,json,hashlib,difflib
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'paper/array_20261104_final_review/elsarticle';NEW=ROOT/'paper/array_20261105_editorial/elsarticle';OUT=ROOT/'reports/ARRAY_EDITORIAL_20261105'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    counts={};figures=0
    for name in ['main','supplement']:
        a=(OLD/(name+'.tex')).read_text(encoding='utf8');b=(NEW/(name+'.tex')).read_text(encoding='utf8')
        tabs=lambda t:re.findall(r'\\begin\{table\}.*?\\end\{table\}',t,re.S)
        assert tabs(a)==tabs(b);counts[name]=len(tabs(b))
        assert re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',a,re.S)==re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',b,re.S)
        labels=re.findall(r'\\label\{([^}]+)\}',b);refs=re.findall(r'\\ref\{([^}]+)\}',b);assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
        for rel in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',b):assert sha(OLD/rel)==sha(NEW/rel);figures+=1
        (OUT/(name+'_editorial.diff')).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='20261104/'+name+'.tex',tofile='20261105/'+name+'.tex')),encoding='utf8')
    assert counts==dict(main=8,supplement=45)
    m=(NEW/'main.tex').read_text(encoding='utf8')
    assert 'pending' not in m.lower() and 'AGERS' not in m and 'agers2026' not in m
    assert 'OpenAI Codex' in m and 'generative AI' in m
    assert not re.search(r'all authors (?:have |had )?approved|all authors have agreed',m,re.I)
    oldbib=(OLD/'references.bib').read_text();newbib=(NEW/'references.bib').read_text()
    assert newbib.strip()==oldbib[:oldbib.index('\n@misc{agers2026,')].strip()
    cited={k for group in re.findall(r'\\cite\{([^}]+)\}',m) for k in group.split(',')};assert len(cited)==25
    for label in ['history','ensemble','benign','buffer','runtime','faith']:assert 'tab:'+label in re.findall(r'\\ref\{([^}]+)\}',m)
    disclosure=(ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_20261105.md').read_text(encoding='utf8');assert '1571374892' in disclosure and 'substantive historical-result overlap' in disclosure
    report=dict(status='PASS_EDITORIAL_ARTIFACT_COMPARISON',unchanged_main_tables=8,unchanged_supplement_tables=45,unchanged_figures=figures,unchanged_equations=True,scientific_references_retained=25,main_agers_narrative_moved=True,editor_disclosure_preserved=True,pending_text_removed=True,approval_claim_added=False,ai_assistance_disclosure_retained=True,new_model_loads=0,new_inference=0,new_fits=0,turnitin_status='NOT_RUN_BROWSER_URL_VERIFICATION_BLOCKED',turnitin_score=None)
    OUT.mkdir(exist_ok=True);(OUT/'artifact_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
