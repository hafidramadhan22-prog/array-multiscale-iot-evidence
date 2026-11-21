"""Readable vector figures regenerated from saved, unchanged numerical evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Polygon
ROOT=Path(__file__).resolve().parents[1]


def polish(base):
    figdir=base/'figures';rep=ROOT/'reports/ARRAY_MAJOR_REVISION_20261103'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    f=pd.read_csv(rep/'factorial_summary.csv');history=pd.read_csv(rep/'history_summary.csv');rt=pd.read_csv(ROOT/'runs/ARRAY_RUNTIME_FULL_20261103_01/replay_raw.csv');shares=pd.read_csv(rep/'xai_shares.csv')
    def save(fig,name):fig.savefig(figdir/(name+'.pdf'));fig.savefig(figdir/(name+'.png'),dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.7))
    for ax,protocol in zip(axs,['temporal','signature']):
        g=f.loc[(f.protocol==protocol)&(f.method=='weighted_voting')].set_index('view').loc[['c40_o1','c40_o10','c60_o1','c60_o10']]
        vals=100*g.macro_f1;ax.bar(range(4),vals,color=['#8798a4','#aebac2','#236b8e','#50a6a0'])
        if protocol=='temporal':ax.errorbar(range(4),vals,yerr=100*g.macro_f1_sd,fmt='none',ecolor='black',capsize=3)
        for i,v in enumerate(vals):ax.text(i,v+(.18 if protocol=='temporal' else .7),f'{v:.3f}',ha='center',fontsize=10.7)
        ax.set_ylim(95 if protocol=='temporal' else 65,100);ax.set_xticks(range(4),['C40/O1','C40/O10','C60/O1','C60/O10'],rotation=40,ha='right',fontsize=11)
        ax.set_ylabel('Macro-F1 (%)',fontsize=12);ax.set_title('Temporal mean' if protocol=='temporal' else 'Signature diagnostic',fontsize=11.5)
    fig.tight_layout(pad=.6);save(fig,'revision_factorial')
    full=f.query("protocol=='signature' and view=='c60_o10' and method=='weighted_voting'").iloc[0];h=history[history.method=='weighted_voting'].set_index('history');cols=['recall_'+c for c in ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']]
    fig,ax=plt.subplots(figsize=(7.2,3.25));xx=np.arange(5)
    for i,(name,row,color) in enumerate([('Full shared',full,'#245b80'),('Shared density',h.loc['shared_density'],'#cf8d49'),('Packet private',h.loc['exclusive'],'#51a29b')]):ax.bar(xx+(i-1)*.25,row[cols].astype(float)*100,.25,label=name,color=color)
    ax.set_xticks(xx,['Normal','DDoS','Scanning','SQL','Ransom'],fontsize=11);ax.set_ylabel('Recall (%)');ax.set_ylim(0,110);ax.legend(loc='lower left',fontsize=11,framealpha=.95);fig.tight_layout(pad=.6);save(fig,'revision_history')
    pivot=rt.pivot(index='repeat',columns='mode',values='total_seconds');fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
    for mode,label,color in [('native_certified','Native certified','#a46539'),('compact_certified','Compact certified','#216f94')]:axs[0].plot(pivot.index+1,pivot[mode],'.-',lw=.8,label=label,color=color)
    axs[0].set(xlabel='Pair',ylabel='Replay time (s)',ylim=(.5,2.9));axs[0].legend(fontsize=10.8,loc='upper right')
    delta=pivot.native_certified-pivot.compact_certified;axs[1].bar(pivot.index+1,delta,color=np.where(delta>0,'#216f94','#b35f4a'));axs[1].axhline(0,color='black',lw=.8);axs[1].set(xlabel='Pair',ylabel='Native - compact (s)');fig.tight_layout(pad=.6);save(fig,'revision_runtime')
    g=shares.query("background=='A32' and method=='weighted_voting'").set_index('group');values=[100*g.loc[k,'share'] for k in ['history_100ms','history_1000ms','history_10000ms']]+[100*g.loc[~g.index.str.startswith('history'),'share'].sum()]
    fig,ax=plt.subplots(figsize=(7.2,3.2));ax.barh(['0.1 s history (20)','1 s history (20)','10 s history (20)','Six protocol groups'],values,color=['#78a8b8','#3e8a9f','#245b80','#a6acaf']);ax.invert_yaxis();ax.set(xlabel='Mean absolute attribution share (%)',xlim=(0,100))
    for i,v in enumerate(values):ax.text(v+.6,i,f'{v:.2f}%',va='center',fontsize=12)
    fig.tight_layout(pad=.6);save(fig,'revision_xai')
    # Study flow: fewer words per box, larger type at final column width.
    fig,ax=plt.subplots(figsize=(9.4,6));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,text,color='#eaf1f6',fs=15):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#31546e',linewidth=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
    def arrow(a,b,label=None):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#31546e','lw':1.3})
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,fontsize=13,ha='center')
    box(.3,5.85,9.4,.85,'Edge development: prior observations retained\nA: experts; G: weights; C: order; V: reused evaluation')
    for x,text in [(.3,'Representation (RQ1)\nC40/C60 x O1/O10\nHistory ownership'),(3.6,'Execution (RQ2)\nFixed f: label certificate\nShared preprocessing'),(6.9,'Explanation (RQ3)\nFull probabilities of f\nWindow/donor audits')]:
        box(x,3.98,2.8,1.12,text,fs=14.5);arrow((x+1.4,5.76),(x+1.4,5.2))
    arrow((6.48,4.53),(6.78,4.53))
    box(.3,2,4.35,1.12,'Observed alarms: payload tests\nGuarded predictor g, selected later\nExecution/XAI of f do not transfer','#fff1e7',14.5)
    box(5.3,2,4.4,1.12,'Frozen external test: f, RF, g\nCIC: capture-scenario labels\nTON: mapped flow labels','#e6f2ef',14.5)
    arrow((1.7,3.88),(1.7,3.22));arrow((4.78,2.55),(5.16,2.55));arrow((5,3.86),(7.1,3.24))
    box(.3,.2,9.4,.9,'Within-trace context gains; same-f execution and explanation checks.\nExternal limits reported separately; no pooled five-class score.',fs=15)
    arrow((2.4,1.9),(2.4,1.2));arrow((7.5,1.9),(7.5,1.2));fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);save(fig,'revision_flow')
    # Earlier payload-assisted h, kept distinct from the later GV g.
    fig,ax=plt.subplots(figsize=(8.93,4.232));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def rect(x,y,w,h,t,color='#eaf1f6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor='#31546e'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=14)
    def arr(a,b,t=None):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.2,'color':'#31546e'})
        if t:ax.text((a[0]+b[0])/2+.02,(a[1]+b[1])/2+.03,t,fontsize=13)
    rect(.02,.83,.96,.14,'Original five-expert voting f and full RF posterior r')
    rect(.03,.43,.42,.25,'f predicts SQL; r predicts Normal;\nRF Normal probability >= 0.5?')
    rect(.55,.43,.42,.25,'Usable content supports SQL\nor gives a binary tie?')
    rect(.03,.13,.42,.16,'Retain full original\nvoting posterior f')
    rect(.55,.13,.42,.16,'Select complete\nRF posterior r','#e6f2ef')
    arr((.24,.82),(.24,.70));arr((.46,.56),(.53,.56),'Yes');arr((.24,.42),(.24,.31),'No');arr((.76,.42),(.76,.31),'No');arr((.59,.42),(.39,.31),'Yes')
    ax.text(.5,.035,'Historical payload rule h; distinct from later guarded voting g.',ha='center',fontsize=13.5);fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);save(fig,'fpr_reconciliation')
    # Two actual audit records; split byte prefixes instead of shrinking the type.
    rows=[json.loads(s) for s in (ROOT/'reports/PAYLOAD_EVIDENCE_20261105_01/actual_payload_traces.jsonl').read_text('utf8').splitlines()];normal=next(r for r in rows if r['row_id']==105044);sql=next(r for r in rows if r['row_id']==23740)
    assert normal['extracted_content']=='24.68 76.42' and sql['extracted_content']=='2 Submitid=53 ORDER BY 1-- MEbL'
    fig,ax=plt.subplots(figsize=(7.2,4.6836));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def tracebox(x,y,w,h,title,text,mono=False,color='#edf3f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#486476'))
        ax.text(x+.015,y+h-.02,title,ha='left',va='top',fontsize=12,fontweight='bold',color='#153b50');ax.text(x+.015,y+h-.08,text,ha='left',va='top',fontsize=11.2,fontfamily='DejaVu Sans Mono' if mono else 'DejaVu Sans',linespacing=1.25)
    for x,r,title in [(.025,normal,'Normal | row 105044'),(.525,sql,'SQL | row 23740')]:
        ax.text(x,.98,title,fontsize=12.5,fontweight='bold',va='top');parts=[r['tcp_payload_hex'][i:i+2] for i in range(0,24,2)];prefix=' '.join(parts[:6])+'\n'+' '.join(parts[6:])
        tracebox(x,.70,.45,.20,'TCP payload: hex prefix',prefix,True)
        decoded='MQTT measurement\nTemperature_and_Humidity' if r is normal else 'HTTP GET\n/DVWA/vulnerabilities/\nsqli_blind/'
        tracebox(x,.40,.45,.21,'Decoded current packet',decoded)
        content=r['extracted_content'] if r is normal else '2 Submitid=53\nORDER BY 1-- MEbL';tracebox(x,.13,.45,.18,'Classifier content',content,True,'#e4f0eb')
        for a,b in [((x+.225,.69),(x+.225,.62)),((x+.225,.39),(x+.225,.32))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#486476'})
    ax.text(.5,.045,'Different protocols; not a matched benign/malicious HTTP pair.',ha='center',fontsize=11.2);ax.text(.5,.005,'Headers excluded. Full bytes retained in evidence. No request executed.',ha='center',fontsize=10.8);fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);save(fig,'payload_trace_audit')
    # FPR panels stacked so the numeric annotations remain readable in print.
    cached=pd.read_csv(ROOT/'reports/FPR_SOLUTION_20261109/metrics_all_cached.csv');later=pd.read_csv(ROOT/'reports/FPR_SOLUTION_20261109/metrics_later_blocks.csv');external=pd.read_csv(ROOT/'reports/INDEPENDENT_CAPTURE_20261109/primary_policy_metrics_verified.csv');K='sql_guard_rf_rescue'
    left=[]
    for policy in ['weighted_reference',K]:left.append([cached.loc[(cached.cohort=='additional760')&(cached.policy==policy),'FPR_percent'].iloc[0]]+[later.loc[(later.block==b)&(later.policy==policy),'FPR_percent'].iloc[0] for b in [1,2,3]]+[cached.loc[(cached.cohort=='V_seed42')&(cached.policy==policy),'FPR_percent'].iloc[0]])
    right=[[external.loc[(external.source==source)&(external.policy==p),'FPR_percent'].iloc[0] for source in ['CIC_Normal0','CIC_Normal3','TON_Ransom1','TON_Ransom2']] for p in ['weighted_reference',K]]
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.45))
    for ax,values,names,title,top in [(axs[0],left,['Initial\nn=760','Block 1\nn=769','Block 2\nn=699','Block 3\nn=697','V Normal\nn=3,860'],'(a) Edge within-trace cohorts',85),(axs[1],right,['CIC N0','CIC N3','TON 1*','TON 2*'],'(b) External Normal targets',55)]:
        xx=np.arange(len(names))
        for shift,ys,label,color in [(-.18,values[0],'Reference WV','#b66441'),(.18,values[1],'Guarded voting','#147d82')]:
            ax.bar(xx+shift,ys,.33,label=label,color=color)
            for x,y in zip(xx+shift,ys):ax.text(x,y+.6,f'{y:.2f}' if y<2 else f'{y:.1f}',ha='center',fontsize=11)
        ax.set_xticks(xx,names,fontsize=11);ax.set_ylabel('FPR (%)',fontsize=12);ax.set_title(title,loc='left',fontsize=12);ax.set_ylim(0,top)
    axs[1].axhline(8,color='#a4293c',ls='--',lw=1,label='External target: 8%');axs[0].legend(fontsize=11,ncol=2,loc='upper right',frameon=False);axs[1].legend(fontsize=10.8,loc='upper right',frameon=False);fig.tight_layout(pad=.6,h_pad=1.2);save(fig,'revision_fpr_transfer')
    inputs=['reports/ARRAY_MAJOR_REVISION_20261103/factorial_summary.csv','reports/ARRAY_MAJOR_REVISION_20261103/history_summary.csv','runs/ARRAY_RUNTIME_FULL_20261103_01/replay_raw.csv','reports/ARRAY_MAJOR_REVISION_20261103/xai_shares.csv','reports/PAYLOAD_EVIDENCE_20261105_01/actual_payload_traces.jsonl']
    (ROOT/'reports/ARRAY_FINAL_INTEGRATION_20261110/figure_input_provenance.json').write_text(json.dumps([dict(path=x,sha256=hashlib.sha256((ROOT/x).read_bytes()).hexdigest()) for x in inputs],indent=2)+'\n',encoding='utf8')
