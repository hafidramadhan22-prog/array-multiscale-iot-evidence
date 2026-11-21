"""Validate the condensed article against parent tables and saved predictions only."""
from pathlib import Path
import hashlib, json, re
import numpy as np
from build_array_concise_20261110 import KEEP, MOVED, tables, table_map, body

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"paper/array_20261110_concise/elsarticle"
OLD = ROOT/"paper/array_20261110_integrated/elsarticle"
OUT = ROOT/"reports/ARRAY_CONCISE_20261110"
K = "sql_guard_rf_rescue"
POLICIES = ["weighted_reference", "random_forest", K]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text("utf8"))

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return dict(z)

def cm(y,p):
    a=np.zeros((5,5),dtype=np.int64)
    np.add.at(a,(y,p.argmax(1)),1)
    return a

def gv(z):
    f=z["weighted_reference"]; r=z["random_forest"]
    a=f.argmax(1); b=r.argmax(1)
    use=((a==3)&(b==0))|((a==0)&(b!=0))
    out=f.copy();out[use]=r[use]
    return out

def main():
    main=(BASE/"main.tex").read_text("utf8")
    sup=(BASE/"supplement.tex").read_text("utf8")
    before=(OLD/"main.tex").read_text("utf8")
    sup0=(OLD/"supplement.tex").read_text("utf8")
    parent=read(ROOT/"reports/ARRAY_FINAL_INTEGRATION_20261110/verification.json")
    assert parent["status"]=="PASS_TABLES_COUNTS_REFERENCES_AND_RECOMPUTED_EVIDENCE"
    assert sha(OLD/"main.tex")==parent["main_source_sha256"]
    assert sha(OLD/"supplement.tex")==parent["supplement_source_sha256"]
    assert len(tables(main))==7 and len(tables(sup))==104
    oldtabs=table_map(before);newtabs=table_map(main)
    for key in KEEP:
        assert body(oldtabs[key])==body(newtabs[key]),key
    for key in MOVED:
        block=next(t for t in tables(sup) if r"\label{relocated:"+key+"}" in t)
        assert body(block)==body(oldtabs[key]),key
    for a,b in zip(tables(sup0),tables(sup)):
        assert body(a)==body(b)
    labels_by_doc={}
    for name,src in [("main",main),("supplement",sup)]:
        labels=re.findall(r"\\label\{([^}]+)",src)
        refs=re.findall(r"\\(?:ref|ref\*)\{([^}]+)",src)
        assert len(labels)==len(set(labels)),(name,"duplicate labels")
        assert set(refs)<=set(labels),(name,set(refs)-set(labels))
        labels_by_doc[name]=labels
        assert "AGERS" not in src and "author approval pending" not in src.lower()
        # "placeholder" is also a legitimate description of absent source-payload cells.
        assert "unused_cross_document_placeholder" not in src and "%%TABLE_" not in src and "%%FIGURE_" not in src
    for key in [*KEEP,"transfer"]:
        assert r"\ref{tab:"+key+"}" in main
    for key in ["flow","factorial","runtime","xai","fpr_transfer"]:
        assert r"\ref{fig:"+key+"}" in main
    bib=(BASE/"references.bib").read_text("utf8")
    assert sha(BASE/"references.bib")==sha(OLD/"references.bib")
    keys=set(re.findall(r"@\w+\{([^,]+),",bib))
    used=set(k for s in re.findall(r"\\cite\{([^}]+)",main) for k in s.split(","))
    assert keys==used and len(keys)==30,(keys-used,used-keys)
    supp_used=set(k for s in re.findall(r"\\cite\{([^}]+)",sup) for k in s.split(","))
    assert supp_used<=keys
    # Prepared posterior arrays are inspected as data; no estimator is loaded.
    external=ROOT/"runs/INDEPENDENT_CAPTURE_20261109_01"
    sources=["CIC_Normal0","CIC_Normal3","CIC_SQL","CIC_PortScan","CIC_DDoS_SYN","TON_Ransom1","TON_Ransom2"]
    cmat={}
    files=[]
    for source in sources:
        p=external/source/"probabilities.npz"
        z=load(p);y=z.get("y_scenario",z.get("y"))
        np.testing.assert_allclose(gv(z),z[K],rtol=0,atol=1e-12)
        for policy in POLICIES:
            cmat[(source,policy)]=cm(y,z[policy])
        files.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    combined={}
    for family,ss in [("CIC",["CIC_Normal0","CIC_Normal3"]),("TON",["TON_Ransom1","TON_Ransom2"])]:
        combined[family]=[sum(cmat[(s,p)] for s in ss) for p in POLICIES]
    rows=[]
    for family in ["CIC","TON"]:
        counts=[int(c[0,1:].sum()) for c in combined[family]]
        supports=[int(c[0].sum()) for c in combined[family]]
        assert supports==[2000,2000,2000]
        rows.append([family+" Normal","2000","FPR",*[f"{100*n/2000:.2f}" for n in counts]])
    for source,title,cl in [("CIC_SQL","CIC SQL",3),("CIC_PortScan","CIC Scan",2),("CIC_DDoS_SYN","CIC SYN",1)]:
        rows.append([title,"1000","Scenario match",*[f"{100*cmat[(source,p)][cl,cl]/1000:.2f}" for p in POLICIES]])
    rows.append(["TON Ransomware","2000","Type recall",*[f"{100*c[4,4]/2000:.2f}" for c in combined["TON"]]])
    transfer=newtabs["transfer"]
    for row in rows:
        assert " & ".join(row)+r"\\" in transfer,row
    assert int(combined["TON"][2][4,0])==1657
    assert int(combined["TON"][0][4,0])==1630
    assert int(combined["TON"][2][4,4])==31
    env=read(ROOT/"runs/ARRAY_RUNTIME_FULL_20261103_01/environment.json")
    assert env["logical_cpus"]==32 and round(env["physical_ram_mib"],1)==16103.9
    assert env["cpu"]=="Intel64 Family 6 Model 183 Stepping 1, GenuineIntel"
    for k,version in [("scikit-learn","1.7.2"),("xgboost","3.0.5"),("lightgbm","4.6.0"),("shap","0.48.0")]:
        assert env["packages"][k]==version and version in main
    v=load(ROOT/"runs/FPR_SQL_GUARD_20261109_01/posteriors/V_seed42.npz")
    vcm=[cm(v["y"],v[p]) for p in POLICIES]
    assert [int(c[0,1:].sum()) for c in vcm]==[36,38,39]
    assert [int(c[0].sum()) for c in vcm]==[3860]*3
    later=[load(p) for p in sorted((ROOT/"runs/FPR_SQL_GUARD_20261109_01/posteriors").glob("*_later.npz"))]
    y=np.concatenate([z["y"] for z in later])
    assert len(y)==697 and not y.any()
    for p,fp in zip(POLICIES,[396,1,1]):
        c=cm(y,np.concatenate([gv(z) if p==K else z[p] for z in later]))
        assert c[0,1:].sum()==fp
    for term in ["1.1425","1.1475","41.75", "0.1435","1.0104","31.90","18.85","1,657/2,000"]:
        assert term in main,term
    assert r"\section*{Declaration of generative AI" in main
    # Confirm the six historical figures and all new-source figure bytes are retained.
    for image in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)",main+sup):
        assert (BASE/image).is_file(),image
        assert sha(BASE/image)==sha(OLD/image),image
    result=dict(status="PASS_CONDENSED_MANUSCRIPT_AND_SAVED_EVIDENCE",
                main_source_sha256=sha(BASE/"main.tex"),supplement_source_sha256=sha(BASE/"supplement.tex"),
                main_tables=7,main_figures=5,supplement_tables=104,supplement_figures=9,
                parent_main_table_bodies_preserved=15,parent_supplement_table_bodies_preserved=95,
                relocated_main_tables=9,reference_count=30,new_external_summary_rows_recomputed=rows,
                external_decisions_rechecked=27000,reference_V_Normal_FP=36,guard_V_Normal_FP=39,
                block3_guard_FP=1,block3_scorable=697,external_TON_guard_Ransomware_to_Normal=1657,
                input_arrays=files,new_fits=0,new_inference=0,models_deserialized=False,
                FPR_below1_all_populations=False,external_detection_reliability_established=False)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="input_arrays"},indent=2))

if __name__=="__main__":
    main()
