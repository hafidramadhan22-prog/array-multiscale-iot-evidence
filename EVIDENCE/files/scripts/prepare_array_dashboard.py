"""Create an offline dashboard bundle from verified project-owned artifacts.

No training and no deserialization here. Original runs remain read-only.
"""
from pathlib import Path
import csv, hashlib, json, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from importlib.metadata import version
from run_wb_research import load_data
from multiscale_context import repair_frame


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, record):
    p=(root/record['path']).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():raise ValueError('Invalid artifact path')
    if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Checksum mismatch '+str(p))
    return p


def main():
    dest=ROOT/'dashboard'
    assets=dest/'assets';assets.mkdir(parents=True,exist_ok=True)
    runtime=ROOT/'runs/EXACT_RUNTIME_20261020_01'
    complete=json.loads((runtime/'complete.json').read_text())
    assert complete['status']=='PASS_EXACT_RUNTIME'
    assert complete['inputs_manifest_sha256']==sha(runtime/'inputs_manifest.json')
    f=json.loads((runtime/'inputs_manifest.json').read_text())
    catalog={r['path']:r for r in json.loads((runtime/'artifacts_sha256.json').read_text())}
    for key in ['inputs_manifest.json','verification_independent.json','complete.json']:
        verify(runtime,catalog[key])
    spec=f['policies']['baseline25__compact_certified']
    shutil.copy2(verify(runtime,spec['model']),assets/'model.joblib')
    xai=ROOT/'runs/XAI_MULTISCALE_FULL_20261020_01'
    xc={r['path']:r for r in json.loads((xai/'artifacts_sha256.json').read_text())}
    for rel in ['indices.npz','groups/A32/coalitions.npz','groups/A32/metadata.json','configuration_freeze.json']:
        verify(xai,xc[rel])
    with np.load(xai/'indices.npz',allow_pickle=False) as z:idx={k:z[k].copy() for k in z.files}
    cfg=json.loads((ROOT/'runs/MULTISCALE_REPAIR_FULL_20261020_01/configuration_freeze.json').read_text())
    raw,y,holdout,data=load_data(cfg)
    X,_,_=repair_frame(raw,data,'legacy25_multiscale60')
    assert not np.intersect1d(idx['row_ids'],holdout).size
    assert list(X.columns)==f['schemas']['baseline25']['columns']
    X.loc[idx['row_ids']].to_csv(assets/'samples.csv',index_label='row_id',float_format='%.17g')
    classes=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
    with np.load(xai/'groups/A32/coalitions.npz',allow_pickle=False) as z:
        values=z['values'];phi=z['phi'];np.testing.assert_array_equal(z['row_ids'],idx['row_ids'])
    meta=json.loads((xai/'groups/A32/metadata.json').read_text())
    samples=[]
    for i,row_id in enumerate(idx['row_ids']):
        p=values[i,-1,0];c=int(p.argmax())
        samples.append(dict(row_id=int(row_id),true_label=classes[int(idx['y'][i])],
            full_reference=p.tolist(),shap=dict(background='A32',predicted_class=classes[c],
                base_probability=float(values[i,0,0,c]),full_probability=float(p[c]),
                groups=list(meta['groups']),values=phi[i,:,0,c].tolist(),
                source='Stored exact grouped Shapley audit; not recomputed for uploaded inputs.')))
    (assets/'samples.json').write_text(json.dumps(samples,indent=2)+'\n',encoding='utf8')
    replay=assets/'replay';replay.mkdir(exist_ok=True)
    needed=['inputs/baseline25_1000.csv','inputs/baseline25_replay.csv','inputs/expected.npz','replay/workload.json']
    files={r['path']:r for r in f['files']}
    for rel in needed:
        shutil.copy2(verify(runtime,files[rel]),assets/Path(rel).name)
    work=json.loads((assets/'workload.json').read_text())
    for item in work['files']:
        rel='replay/'+Path(item['file']).name
        shutil.copy2(verify(runtime,files[rel]),replay/Path(rel).name)
    (assets/'schema.json').write_text(json.dumps(f['schemas']['baseline25'],indent=2)+'\n',encoding='utf8')
    def readcsv(path):
        return list(csv.DictReader(path.open(encoding='utf8',newline='')))
    paper=ROOT/'paper/20261020'
    results=dict(
        historical=dict(protocol='Historical holdout; exploratory six-configuration comparison',
            rows=readcsv(ROOT/'reports/REKOMENDASI_20261015_01/ranking_all_six.csv')),
        representation=dict(protocol='Temporal development means over three estimator seeds; signature is a separate diagnostic',
            rows=readcsv(paper/'quality_multiscale_all_treatments.csv')),
        runtime=dict(protocol='One laptop session; fixed workload; not live request timings',
            rows=readcsv(paper/'runtime_summary.csv')),
        faithfulness=dict(protocol='Stored 100-target class-balanced audit; rankings A32, donors B32',
            rows=readcsv(paper/'faithfulness.csv')),
        robustness=dict(protocol='Stored query-delay perturbation; changed-input subset reported separately',
            rows=readcsv(paper/'robustness.csv')),
    )
    (assets/'research.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    sources=dest/'core';sources.mkdir(exist_ok=True)
    records=[]
    for name in ['exact_runtime.py','wb_scheduler.py','ta_hafid_engine.py','sbfe.py','wb_behavior.py','multiscale_context.py','e4.py']:
        source=ROOT/'src'/name;shutil.copy2(source,sources/name)
        records.append(dict(path='src/'+name,sha256=sha(source)))
    packages=['numpy','pandas','scipy','scikit-learn','imbalanced-learn','xgboost','lightgbm','joblib','psutil','threadpoolctl','matplotlib']
    (dest/'requirements.txt').write_text('\n'.join(p+'=='+version(p) for p in packages)+'\n',encoding='utf8')
    origin=dict(model_source='runs/EXACT_RUNTIME_20261020_01/'+spec['model']['path'],
        model_sha256=spec['model']['sha256'],runtime_catalog_sha256=sha(runtime/'artifacts_sha256.json'),
        xai_catalog_sha256=sha(xai/'artifacts_sha256.json'),source_records=records,
        classifier_fits=0,only_owned_model=True,classes=classes,
        reference='WB25 + multiscale60 Balanced; temporal seed 42; G-selected voting reference',
        limitations=['Prepared 85-field inputs; raw 25-field extraction is not implemented.',
            'Sample and saved explanations reuse development V, not a new test.',
            'Replay retains supplied legacy packet fields; not a live network sensor.'])
    (assets/'origin.json').write_text(json.dumps(origin,indent=2)+'\n',encoding='utf8')
    # Anchor the asset manifest in source; input uploads cannot replace trusted models.
    allfiles=sorted([p for p in assets.rglob('*') if p.is_file()]+list(sources.glob('*.py')))
    manifest=[dict(path=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in allfiles]
    mp=dest/'assets_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    (dest/'trusted_assets.py').write_text('MANIFEST_SHA256 = '+repr(sha(mp))+'\n',encoding='utf8')
    print(json.dumps(dict(status='DASHBOARD_ASSETS_VERIFIED',files=len(manifest),sample_rows=100,
        numeric_features=len(f['schemas']['baseline25']['numeric']),categorical_features=len(f['schemas']['baseline25']['categorical']),
        model_sha256=spec['model']['sha256'],no_training=True,no_deserialization=True),indent=2))


if __name__=='__main__':main()
