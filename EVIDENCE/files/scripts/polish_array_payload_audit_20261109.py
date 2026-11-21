"""Final source-level typography/wording corrections, with versioned diff refresh."""
from pathlib import Path
import json,difflib,re
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'paper/array_20261109_payload_audit/elsarticle'
OLD=ROOT/'paper/array_20261105_fpr_update/elsarticle'
R=ROOT/'reports/ARRAY_PAYLOAD_AUDIT_REVISION_20261109'
for n in ['main','supplement']:
    p=B/(n+'.tex');s=p.read_text('utf8')
    s=s.replace('The payload audit follows each selected record from its stored TCP/MQTT hexadecimal fields to decoded current content, twelve syntax values and saved classifier output. Canonical row identifiers and source CSV/packet indices are kept distinct.',
                'The audit follows each record from the stored TCP payload or MQTT message through decoding and feature extraction to its classifier output. Canonical row identifiers and the source CSV and packet indices are kept separate.')
    s=s.replace(' This declaration does not imply that an AI-detection or similarity score has been measured.','')
    s=s.replace('TF-IDF and hybrid content classifiers predict Normal.','TF-IDF and hybrid weighted content classifiers predict Normal.')
    s=s.replace(' Table S73 records the resulting interpretation without another fit, threshold change or method-selection step.','')
    s=s.replace(r'\setlength{\bibsep}{4pt}',r'\setlength{\bibsep}{3pt}\interlinepenalty=10000')
    s=s.replace('The prototype passed 10 unit/integration, 16 HTTP and 20 virtual-DOM checks, with a separate isolated-Python package check. Graphical inspection was interrupted, so these results do not establish visual quality or usability. Interactive request timings are distinct from repeated benchmark results. Manifests retain source hashes, splits, seeds, training-only resampling counts, choices, predictions, games and raw resource repeats. The earlier research snapshot records 170 tests; these are software checks, not scientific replication.',
                'The prototype passed 10 unit/integration, 16 HTTP and 20 virtual-DOM checks, together with an isolated-Python package check. These automated checks cover implementation behavior; they do not establish interface usability. Interactive request timings are kept separate from the repeated benchmarks. Manifests retain source hashes, splits, seeds, training-only resampling counts, selection decisions, predictions, donor games and raw resource measurements. The earlier snapshot records 170 software tests, which likewise do not constitute scientific replication.')
    s=s.replace('Table 1 states the narrower distinction.',r'Table~\ref{tab:related_short} states the narrower distinction.')
    marker='Bracketed hexadecimal values denote control bytes. '
    if n=='supplement' and marker not in s:
        s=s.replace('The value 0 in a placeholder field is not decoded as a one-byte payload.',marker+'The value 0 in a placeholder field is not decoded as a one-byte payload.')
    s=re.sub(r'(?:Bracketed hexadecimal values denote control bytes\. )+',marker,s)
    s=''.join(c if ord(c)>=32 or c in '\n\r\t' else '[0x%02x]'%ord(c) for c in s)
    p.write_text(s,encoding='utf8')
    (R/(n+'_changes.diff')).write_text(''.join(difflib.unified_diff((OLD/(n+'.tex')).read_text('utf8').splitlines(True),s.splitlines(True),fromfile='5 November/'+n,tofile='9 November/'+n)),encoding='utf8')
p=R/'table_ledger.json';ledger=json.loads(p.read_text('utf8'))
for t in ledger:
    t['rows']=[[''.join(c if ord(c)>=32 else '[0x%02x]'%ord(c) for c in str(v)) for v in row] for row in t['rows']]
    if t['label']=='sup:actual_payload_rows':
        if marker not in t['note']:t['note']=t['note'].replace('The value 0 in a placeholder field is not decoded as a one-byte payload.',marker+'The value 0 in a placeholder field is not decoded as a one-byte payload.')
        t['note']=re.sub(r'(?:Bracketed hexadecimal values denote control bytes\. )+',marker,t['note'])
    if t['label']=='sup:audit_decisions':
        t['rows'][-1]=['Versioned interpretation','Preserve the numerical record and retain the failed probes alongside the previously reported alarm reduction.']
        t['note']=r'This ledger records interpretation after the 5 November model experiments and the 6 November payload audit. No new dataset, capture, learner, fitting stage or threshold selection was added to prepare this document revision.'
p.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('SOURCE_POLISH_COMPLETE')
