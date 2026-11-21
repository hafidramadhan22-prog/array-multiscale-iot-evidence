"""Read-only scientific rechecks routed to new reports; never load estimators or fit."""
from pathlib import Path
import sys,json,hashlib,shutil,importlib,ast
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
REPORT=ROOT/'reports/ARRAY_REVISION_20261117'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def run(which):
    REPORT.mkdir(parents=True,exist_ok=True)
    configs={
      'core':('verify_array_revision_20261103','OUT','main'),
      'integration':('verify_array_final_integration_20261110','OUT','main'),
      'concise':('verify_array_concise_20261110','OUT','main'),
      'external':('audit_independent_results_20261109','OUT','audit'),
      'guard':('verify_fpr_solution_20261109','OUT','main'),
      'payload':('verify_array_fpr_update_20261105','OUT','main')}
    name,key,entry=configs[which];m=importlib.import_module(name);old=getattr(m,key)
    dest=REPORT/'saved_evidence_checks'/which;dest.mkdir(parents=True,exist_ok=True)
    # Certain verifiers consume their original report-side ledgers/reference metadata.
    if which in ['integration','concise','payload']:
        shutil.copytree(old,dest,dirs_exist_ok=True)
    setattr(m,key,dest);getattr(m,entry)()
    print('PASS_RECHECK_'+which.upper(),flush=True)

def coverage():
    import pandas as pd,numpy as np
    REPORT.mkdir(parents=True,exist_ok=True)
    canonical=ROOT/'ML-EdgeIIoT-dataset.csv'
    assert sha(canonical)=='53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a'
    raw=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01/TON_flow_mapping/mapping_complete.json'
    mapping=json.loads(raw.read_text('utf8'));saved=pd.read_csv(ROOT/'reports/INDEPENDENT_CAPTURE_20261109/TON_label_coverage.csv').set_index('source')
    records=[]
    for src in mapping['sources']:
        c=src['mapping_counts'];total=int(src['total_packets']);n=c['normal']+c['ransomware'];warm=c['warmup']
        assert sum(c.values())==total
        expected=100*n/total;after=100*n/(total-warm)
        assert abs(expected-saved.loc[src['name'],'in_scope_labelled_percent'])<1e-12
        records.append(dict(source=src['name'],all_packets=total,normal=c['normal'],ransomware=c['ransomware'],warmup=warm,
          denominator_in_archived_formula=total,all_packet_mapped_percent=expected,post_warmup_denominator=total-warm,
          post_warmup_mapped_percent_for_audit_only=after,existing_display_percent=f'{expected:.4f}'))
    pd.DataFrame(records).to_csv(REPORT/'S90_denominator_audit.csv',index=False)
    assert [r['existing_display_percent'] for r in records]==['62.9036','6.6602']
    history=pd.read_csv(ROOT/'reports/ARRAY_MAJOR_REVISION_20261103/history_summary.csv')
    exclusive=history[history.history=='exclusive'].set_index('method')
    factorial=pd.read_csv(ROOT/'reports/ARRAY_MAJOR_REVISION_20261103/factorial_summary.csv')
    temporal=factorial[(factorial.protocol=='temporal')&(factorial.view=='c60_o10')].set_index('method')
    numbers=dict(private_WV_macro_F1_percent=100*exclusive.loc['weighted_voting','macro_f1'],
      private_RF_macro_F1_percent=100*exclusive.loc['random_forest','macro_f1'],
      private_RF_minus_WV_pp=100*(exclusive.loc['random_forest','macro_f1']-exclusive.loc['weighted_voting','macro_f1']),
      temporal_WV_macro_F1_percent=100*temporal.loc['weighted_voting','macro_f1'],
      temporal_uniform_macro_F1_percent=100*temporal.loc['uniform_voting','macro_f1'])
    # The existing tabulated full-shared result is recomputed by the core verifier.
    (REPORT/'verified_comparison_numbers.json').write_text(json.dumps(numbers,indent=2)+'\n','utf8')
    module=ROOT/'scripts/audit_independent_results_20261109.py';tree=ast.parse(module.read_text('utf8'))
    assert any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.right,ast.Subscript) and isinstance(n.right.slice,ast.Constant) and n.right.slice.value=='total_packets' for n in ast.walk(tree))
    out=dict(status='PASS_S90_ALL_PACKET_DENOMINATOR_AND_COMPARISONS',coverage=records,comparison=numbers,
      archived_formula_source=str(module.relative_to(ROOT)),archived_formula_source_sha256=sha(module),
      mapping_source_sha256=sha(ROOT/'scripts/map_ton_intervals_fast_20261109.py'),mapping_result_sha256=sha(raw),
      correction='Clarify table definition; retain original values. Warm-up packets excluded from eligible labelled numerator, retained in all-packet denominator.',
      old_run_outputs_unchanged=True,new_fits=0,new_inference=0,models_deserialized=False)
    (REPORT/'denominator_and_comparison_audit.json').write_text(json.dumps(out,indent=2)+'\n','utf8');print(json.dumps(out,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='coverage':coverage()
    else:run(sys.argv[1])
