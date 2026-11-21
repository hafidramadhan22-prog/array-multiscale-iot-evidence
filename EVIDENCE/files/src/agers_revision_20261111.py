"""Shared provenance and stateless conference-pipeline utilities."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k]="1"
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FULL=ROOT/"runs/FULL_20261013_01"
RUN=ROOT/"runs/AGERS_BUDGET_CONTROL_20261111_01"
REPORT=ROOT/"reports/AGERS_CAMERA_READY_20261111"
FULL_CATALOG_SHA="9826fa8870848f5c2ce88b80a44c9f5d2efcc4b3318ce0a6e9d60f429bab4437"
DATA_SHA="53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a"
CLASSES=["Normal","DDoS_TCP","Port_Scanning","SQL_injection","Ransomware"]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text("utf8"))
def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+"\n",encoding="utf8")

def checked(rel):
    catalog=FULL/"artifacts_sha256.json"
    assert sha(catalog)==FULL_CATALOG_SHA,"Parent catalog changed"
    rel=str(rel).replace("\\","/")
    row=next(r for r in read(catalog) if r["path"].replace("\\","/")==rel)
    p=FULL/rel
    assert p.stat().st_size==row["bytes"] and sha(p)==row["sha256"],rel
    return p

def engine():
    p=checked("runner_source/ta_hafid_engine.py")
    spec=importlib.util.spec_from_file_location("agers_frozen_engine_20261111",p)
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);e.configure_features("dstport29")
    return e

def data(e):
    p=ROOT/"ML-EdgeIIoT-dataset.csv"
    assert sha(p)==DATA_SHA
    raw=pd.read_csv(p,low_memory=False)
    assert raw.shape==(157800,63)
    fold=pd.read_csv(checked("dstport29/tables/training_fold_assignment.csv"))
    hold=pd.read_csv(checked("dstport29/tables/holdout_row_ids.csv")).original_row_id.to_numpy()
    ids=fold.original_row_id.to_numpy()
    assert len(ids)==52684 and len(hold)==13171 and not set(ids)&set(hold)
    X=e.deterministic_domain_transform(raw.loc[ids])
    y=raw.loc[ids,"Attack_type"].map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    assert X.shape==(52684,29) and np.array_equal(y,fold.label.to_numpy())
    cfg=read(checked("dstport29/manifests/effective_config.json"))
    cfg["_training_ids"]=ids;cfg["_holdout_ids"]=hold
    return X,y,fold,cfg

def probabilities(b,X):
    kind=b["kind"]
    if kind=="historical":
        v=b["bundle"];ps=[np.asarray(v["models"][n].predict_proba(X)) for n in v["model_order"]]
        if v["method"]=="soft_voting":return np.mean(ps,axis=0)
        if v["method"]=="weighted_voting":return sum(v["weights"][n]*p for n,p in zip(v["model_order"],ps))
        return v["meta"].predict_proba(np.hstack(ps))
    if kind=="singleton":return b["pipeline"].predict_proba(X)
    ps=[]
    for n in b["names"]:
        Z=b["preprocessor"].transform(X)
        ps.append(b["estimators"][n].predict_proba(Z.toarray() if n=="naive_bayes" else Z))
    if kind=="hard3":return sum(np.eye(5)[p.argmax(1)] for p in ps)/3
    return b["meta"].predict_proba(np.hstack(ps))

def owned_load(p,expected):
    assert sha(p)==expected,"Unknown fitted model"
    import joblib
    return joblib.load(p)

def final_policies(e):
    import joblib
    models={}
    for case in ["original","balanced"]:
        manifest=read(checked(f"dstport29/manifests/fit_{case}.json"))
        for method in e.METHODS:
            key=case+"__"+method;p=checked(f"dstport29/models/{key}.joblib")
            models[key]=dict(kind="historical",bundle=owned_load(p,manifest[key]["sha256"]))
        base=models[case+"__weighted_voting"]["bundle"]
        for n in ["random_forest","lightgbm"]:
            models[case+"__"+n]=dict(kind="singleton",pipeline=base["models"][n])
    m=read(RUN/"model_manifest.json")
    for key,row in m.items():
        models[key]=owned_load(RUN/row["path"],row["sha256"])
    return models

def metrics(e,y,p):
    from sklearn.metrics import confusion_matrix
    out=e.metrics(y,p);cm=confusion_matrix(y,p.argmax(1),labels=range(5))
    out.update(n=int(len(y)),errors=int((p.argmax(1)!=y).sum()),
               normal_n=int(cm[0].sum()),false_alarms=int(cm[0,1:].sum()),
               attack_to_Normal=int(cm[1:,0].sum()))
    out["FPR_percent"]=float(100*cm[0,1:].sum()/cm[0].sum()) if cm[0].sum() else None
    for i,c in enumerate(CLASSES):
        out["recall_"+c]=float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None
    return out
