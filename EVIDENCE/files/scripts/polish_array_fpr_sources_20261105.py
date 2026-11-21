"""Layout-only line breaks and prose spacing; retain all numerical table cells."""
from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'paper/array_20261105_fpr_update/elsarticle'
replacements={'and117':'and 117','with3,860':'with 3,860','and6,498':'and 6,498','Another64':'Another 64','of321':'of 321','The257':'The 257','all60':'all 60','Normal>=0.5':'Normal >=0.5','pRF(Normal)>=0.5':'pRF(Normal) >=0.5','of2 cold':'of 2 cold','Latency:10':'Latency: 10','after3 warm-ups':'after 3 warm-ups','with1 ms':'with 1 ms','Supports:3,860':'Supports: 3,860','total10,358':'total 10,358','the107':'the 107','all867':'all 867','0eligible':'0 eligible','All24':'All 24','all24':'all 24','100 and memory12':'100 and memory 12','latency100':'latency 100','runnable760':'runnable 760','separate5 November':'separate 5 November','their68':'their 68','the70 earlier':'the 70 earlier','Normal50MQTT':'Normal 50 MQTT','SQL100HTTP':'SQL 100 HTTP','threshold grid0.5':'threshold grid 0.5','onG withC':'on G with C','afterV/additional':'after V/additional','observing257':'observing 257','fixed0.5':'fixed 0.5','hybridWV':'hybrid WV','andRF-only':'and RF-only','/41.75% replay':'/41.75% replay','andSHAP':'and SHAP','Earlier500-row':'Earlier 500-row','its1/760':'its 1/760','from321/760':'from 321/760','selection20261105':'selection 20261105','OOF59':'OOF 59','estimator42':'estimator 42','max2048':'max 2048','min_df1':'min_df 1','+50 other':'+ 50 other','Twohundred':'Two hundred','additional760':'additional 760'}
def apply(t):
    for a,b in {'has0':'has 0','batch1,000':'batch 1,000','batch760':'batch 760','Python3.11.15':'Python 3.11.15','scikit-learn1.7.2':'scikit-learn 1.7.2','NumPy2.2.6':'NumPy 2.2.6','pandas2.2.3':'pandas 2.2.3','imbalanced-learn0.14.0':'imbalanced-learn 0.14.0','XGBoost3.0.5':'XGBoost 3.0.5','LightGBM4.6.0':'LightGBM 4.6.0','joblib1.5.3':'joblib 1.5.3'}.items():t=t.replace(a,b)
    for a,b in replacements.items():
        t=t.replace(a,b)
        if '_' in a:
            t=t.replace(a.replace('_',r'\_\allowbreak{}'),b.replace('_',r'\_\allowbreak{}'))
    return t
def main():
    for name in ('main','supplement'):
        p=B/(name+'.tex');s=p.read_text(encoding='utf8')
        if name=='supplement':
            start=s.index(r'\clearpage\section{Post-observation payload')
            end=s.index(r'\section{Artifact map and submission checks}')
            a=s[start:end].replace(r'\_\allowbreak{}',r'\_')
            a=a.replace(r'\_',r'\_\allowbreak{}').replace(r'\textbf{Threshold}',r'\textbf{Cutoff}')
            s=s[:start]+apply(a)+apply(s[end:])
        else:s=apply(s)
        p.write_text(s,encoding='utf8')
    p=ROOT/'reports/ARRAY_FPR_UPDATE_20261105/table_ledger.json';ledger=json.loads(p.read_text(encoding='utf8'))
    for r in ledger:
        r['caption']=apply(r['caption']);r['note']=apply(r['note']);r['header']=['Cutoff' if h=='Threshold' else h for h in r['header']];r['rows']=[[apply(c) for c in row] for row in r['rows']]
    p.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('POLISHED_LAYOUT_AND_PROSE')
if __name__=='__main__':main()
