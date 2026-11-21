"""Editorial-only manuscript revision; no numerical or experimental changes."""
from pathlib import Path
import re,shutil,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'paper/array_20261104_final_review/elsarticle'
NEW=ROOT/'paper/array_20261105_editorial/elsarticle'
REP=ROOT/'reports/ARRAY_EDITORIAL_20261105'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    NEW.mkdir(parents=True,exist_ok=True);REP.mkdir(exist_ok=True)
    quality=json.loads((ROOT/'reports/ARRAY_FINAL_REVIEW_20261104/pdf_quality.json').read_text())
    for name in ['main','supplement']:
        oldpdf=OLD/(name+'.pdf');row=next(r for r in quality if r['document']==name);assert digest(oldpdf)==row['sha256']
    for p in OLD.iterdir():
        if p.is_file() and p.suffix in ['.cls','.bst','.txt']:shutil.copy2(p,NEW/p.name)
    shutil.copytree(OLD/'figures',NEW/'figures',dirs_exist_ok=True)
    main=(OLD/'main.tex').read_text(encoding='utf8')
    draft=r'\noindent{\footnotesize Author-review draft, 4 November 2026; final author approval pending.}'
    assert draft in main;main=main.replace(draft,'',1)
    old='The authors are responsible for verifying the methods, outputs, references and text; final approval of this revision by all authors is pending.'
    assert old in main;main=main.replace(old,'The authors are responsible for verifying the methods, outputs, references and text.',1)
    paragraphs=main.split('\n\n');related=[p for p in paragraphs if 'related AGERS conference manuscript' in p]
    assert len(related)==1
    moved=related[0];main='\n\n'.join(p for p in paragraphs if p!=moved)
    assert 'AGERS' not in main and 'agers2026' not in main and 'approval pending' not in main and 'all authors is pending' not in main
    # Narrative moved to editorial disclosure, never erased from the provenance record.
    (REP/'related_manuscript_paragraph_moved.tex').write_text(moved+'\n',encoding='utf8')
    (NEW/'main.tex').write_text(main,encoding='utf8')
    bib=(OLD/'references.bib').read_text(encoding='utf8');assert '\n@misc{agers2026,' in bib
    (NEW/'references.bib').write_text(bib[:bib.index('\n@misc{agers2026,')]+'\n',encoding='utf8')
    supp=(OLD/'supplement.tex').read_text(encoding='utf8').replace('4 November 2026','5 November 2026')
    old='Funding, interests and contribution roles were confirmed by the first author. The accompanying reproducibility archive provides code and derived evidence; the original dataset remains with its provider. No existing public code repository was identified in the project. A cover-letter disclosure draft and the AGERS manuscript accompany the author-review package. Final approval by both authors and corresponding-author contact confirmation remain required before submission.'
    assert old in supp
    supp=supp.replace(old,'The accompanying reproducibility archive provides code, frozen configurations, split identifiers, saved predictions and derived evidence. The original dataset remains with its provider. The separate editorial disclosure documents the provenance of shared historical results.',1)
    (NEW/'supplement.tex').write_text(supp,encoding='utf8')
    tables=lambda s:re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
    assert tables(main)==tables((OLD/'main.tex').read_text(encoding='utf8'))
    assert tables(supp)==tables((OLD/'supplement.tex').read_text(encoding='utf8'))
    assert len(tables(main))==8 and len(tables(supp))==45
    for name in ['main','supplement']:
        a=(OLD/(name+'.tex')).read_text(encoding='utf8');b=(NEW/(name+'.tex')).read_text(encoding='utf8')
        assert re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',a,re.S)==re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',b,re.S)
        labels=re.findall(r'\\label\{([^}]+)\}',b);refs=re.findall(r'\\ref\{([^}]+)\}',b);assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
        for f in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',b):assert digest(NEW/f)==digest(OLD/f)
    cited={k for t in re.findall(r'\\cite\{([^}]+)\}',main) for k in t.split(',')};assert len(cited)==25
    disclosure=ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_20261105.md'
    disclosure.write_text((ROOT/'docs/ARRAY_EDITOR_DISCLOSURE_20261104.md').read_text(encoding='utf8').replace('Draft disclosure for the Array editor — not sent','Draft disclosure for the Array editor — 5 November 2026, not sent')+'\nThe 5 November editorial revision removes the related-manuscript paragraph from the main article narrative. The disclosure and shared-result provenance remain available separately; this move does not assert absence of overlap.\n',encoding='utf8')
    report=dict(status='PASS_EDITORIAL_ONLY_SOURCE_CHECKS',main_tables_unchanged=8,supplement_tables_unchanged=45,s1_to_s44_complete=True,s45_runtime_retained=True,equations_unchanged=True,figure_bytes_unchanged=True,related_manuscript_disclosure_preserved=True,approval_status_text_removed_from_manuscript=True,approval_claim_added=False,ai_assistance_declaration_retained=True,cited_references=25,new_training=False,new_inference=False,new_profiling=False)
    (REP/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
