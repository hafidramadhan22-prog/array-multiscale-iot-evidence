"""Real inference for two-view fusion, reusing checksum-verified development workloads."""
from pathlib import Path
import json,time
import joblib
import numpy as np
import ta_hafid_engine as e
from wb_profile import load_inputs as wb_inputs,owned_bundle,quality_manifest,checked_file,replay as wb_replay,check_labels
from port_fusion import FusionPredictor

POLICIES=['always25','always29','certified25','certified29','fixed50','global','confidence','learned_clean','learned_robust']


def source_run(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text())
    source=run.parents[1]/cfg['source_run']
    source_manifest=json.loads((run/'source_run.json').read_text())
    if e.sha256(source/'artifacts_sha256.json')!=source_manifest['catalog_sha256']:raise ValueError('Source catalog changed')
    return source


def read_inputs(run):
    source=source_run(run);inputs=wb_inputs(source,'legacy29_symmetric__shared')
    folder=run/'stages/temporal_seed42__balanced'
    with np.load(folder/'evaluation.npz',allow_pickle=False) as z:
        ids=z['row_ids'];methods=list(z['methods']);p=z['probabilities'][:,0].copy()
    expected={}
    lookup={int(r):i for i,r in enumerate(ids)}
    for policy in POLICIES:
        method=policy.replace('certified','always') if policy.startswith('certified') else policy
        for workload in ['batch1','batch1000','replay']:
            target=inputs['expected'][workload+'_row_ids'];positions=[lookup[int(r)] for r in target]
            expected[policy+'__'+workload]=p[methods.index(method),positions].argmax(1)
    inputs['fusion_expected']=expected
    return inputs


class Adapter:
    def __init__(self,predictor,columns25):self.predictor=predictor;self.columns25=columns25;self.port_rows=0
    def predict(self,frame29):
        labels=self.predictor.predict((frame29[self.columns25],frame29))
        port_count=len(frame29) if self.predictor.baseline is not None and self.predictor.baseline[0]=='29' else self.predictor.last_escalation
        self.port_rows+=port_count
        return labels


def load_predictor(run,policy):
    if policy not in POLICIES:raise ValueError('Policy not frozen')
    source=source_run(run);done=json.loads((run/'quality_complete.json').read_text())
    if done['status']!='PORT_FUSION_QUALITY_COMPLETE' or e.sha256(run/'controllers_manifest.json')!=done['controllers_manifest_sha256']:raise ValueError('Controllers incomplete or changed')
    b25=b29=None;size=0;baseline=None
    if policy in ['always25','certified25']:
        b25,record=owned_bundle(source,'legacy25_symmetric');size=record['bytes'];spec=dict(kind='constant',alpha=0.)
        if policy=='certified25':baseline=('25','certified')
    elif policy in ['always29','certified29']:
        b29,record=owned_bundle(source,'legacy29_symmetric');size=record['bytes'];spec=dict(kind='constant',alpha=1.)
        if policy=='certified29':baseline=('29','certified')
    else:
        if policy=='fixed50':spec=dict(kind='constant',alpha=.5)
        else:
            manifest=json.loads((run/'controllers_manifest.json').read_text());matches=[r for r in manifest if r['method']==policy and 'temporal_seed42__balanced/' in r['path']]
            if len(matches)!=1:raise ValueError('Missing verified controller')
            record=matches[0];spec=joblib.load(checked_file(run,record));size+=record['bytes']
        requires25=spec['kind']!='constant' or spec['alpha']!=1
        requires29=spec['kind']!='constant' or spec['alpha']!=0
        if spec['kind']=='leaf' and not any(a>0 for a in spec['leaf_alpha'].values()):requires29=False
        if requires25:b25,record=owned_bundle(source,'legacy25_symmetric');size+=record['bytes']
        if requires29:b29,record=owned_bundle(source,'legacy29_symmetric');size+=record['bytes']
    # Schema only; no extra model is loaded to obtain these column names.
    manifest=json.loads((source/'profiling/inputs_manifest.json').read_text())
    columns25=manifest['schemas']['legacy25_symmetric']['columns']
    return Adapter(FusionPredictor(b25,b29,spec,baseline),columns25),size


def measure(run,policy,workload,predictor,inputs,verify=False):
    predictor.port_rows=0
    if workload=='replay':
        result=wb_replay(source_run(run),'legacy29_symmetric__shared',predictor,inputs,verify_features=verify)
        result['policy']=policy
    else:
        t=time.perf_counter();labels=predictor.predict(inputs[workload]);seconds=time.perf_counter()-t
        result=dict(policy=policy,total_seconds=seconds,decision_rows=len(labels),labels=labels)
    check_labels(result['labels'],inputs['fusion_expected'][policy+'__'+workload])
    result['port_branch_rows']=predictor.port_rows
    return result
