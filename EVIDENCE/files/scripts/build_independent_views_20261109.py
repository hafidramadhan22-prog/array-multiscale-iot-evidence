"""Render verified independent results and embed a review-only notebook."""
from pathlib import Path
import base64,io,json,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_fpr_solution_views_20261109 import image_table
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01';OUT=ROOT/'reports/INDEPENDENT_CAPTURE_20261109';KEY='sql_guard_rf_rescue';POLICIES=['weighted_reference','random_forest',KEY];LABELS={'weighted_reference':'Voting lama','random_forest':'RF',KEY:'SQL guard + RF rescue'}
def main():
 assert json.loads((OUT/'independent_verification.json').read_text())['unique_targets']==9000
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True);m=pd.read_csv(OUT/'primary_policy_metrics_verified.csv');pred=pd.read_csv(OUT/'all_predictions_verified.csv',dtype={'row_id':str},low_memory=False);allm=pd.read_csv(OUT/'all_policy_metrics_verified.csv');coverage=pd.read_csv(OUT/'TON_label_coverage.csv')
 fpr=m.query('normal_support>0').pivot(index='source',columns='policy',values='FPR_percent').reindex(['CIC_Normal0','CIC_Normal3','TON_Ransom1','TON_Ransom2']);fig,ax=plt.subplots(figsize=(8.2,3.6));x=np.arange(4)
 for j,(name,color) in enumerate(zip(POLICIES,['#ba603c','#8b99a5','#147c83'])):
  a=fpr[name].to_numpy();ax.bar(x+(j-1)*.25,a,.23,label=LABELS[name],color=color)
  for xx,v in zip(x+(j-1)*.25,a):ax.text(xx,v+.5,f'{v:.1f}',ha='center',fontsize=9)
 ax.axhline(8,color='#b81d32',ls='--',lw=1,label='Target 8%');ax.set_ylim(0,42);ax.set_ylabel('FPR pada Normal (%)');ax.set_xticks(x,['CIC Normal0','CIC Normal3','TON file1*','TON file2*']);ax.legend(loc='upper left',ncol=2,frameon=False);fig.tight_layout();fig.savefig(figdir/'FPR_independent.png',bbox_inches='tight',dpi=180);plt.close(fig)
 primary=m.copy();primary['policy']=primary.policy.map(LABELS);primary['label_agreement_percent']=100*(1-primary.errors/primary.n)
 t1=primary.query('normal_support>0')[['source','policy','normal_support','false_alarms','FPR_percent']]
 t2=primary.query("source.str.startswith('TON')",engine='python')[['source','policy','attack_support','attack_to_Normal','attack_recall_percent','recall_Ransomware']]
 t3=primary.query("source in ['CIC_SQL','CIC_PortScan','CIC_DDoS_SYN']")[['source','policy','n','label_agreement_percent','attack_to_Normal']]
 changes=[]
 for folder in [RUN/s for s in m.source.unique()]:
  with np.load(folder/'probabilities.npz',allow_pickle=False) as z:
   y=z['y'] if 'y' in z.files else z['y_scenario'];v=z['weighted_reference'].argmax(1);p=z[KEY].argmax(1)
   changes.append(dict(source=folder.name,Normal_FP_added=int(((y==0)&(v==0)&(p!=0)).sum()),Normal_FP_corrected=int(((y==0)&(v!=0)&(p==0)).sum()),attack_or_scenario_suppressed=int(((y>0)&(v!=0)&(p==0)).sum()),attack_or_scenario_rescued=int(((y>0)&(v==0)&(p!=0)).sum())))
 change=pd.DataFrame(changes);change.to_csv(OUT/'rule_changes.csv',index=False)
 norm=pred.query("truth=='Normal' and prediction!='Normal'");dist=norm.groupby(['source','policy','prediction']).size().reset_index(name='FP_count');dist.to_csv(OUT/'false_alarm_predicted_classes.csv',index=False)
 expert=allm.query("source in ['CIC_Normal0','CIC_Normal3']").groupby('policy')[['normal_support','false_alarms']].sum().reset_index();expert['FPR_percent']=100*expert.false_alarms/expert.normal_support
 tables={'Normal_FPR':(t1,'FPR eksternal - masing-masing 1.000 target Normal'),'Ransomware_detection':(t2,'TON: deteksi biner dan recall jenis Ransomware (%)'),'CIC_scenario':(t3,'CIC: kesesuaian label skenario; bukan recall paket terkonfirmasi'),'TON_coverage':(coverage,'Cakupan label flow TON; paket unmapped tidak diberi label buatan'),'rule_changes':(change,'Perubahan alarm: target sama, aturan berbeda'),'expert_CIC_Normal':(expert,'Saved expert diagnostics pada 2.000 Normal CIC; tidak memilih ulang model')}
 for stem,(table,title) in tables.items():table.to_csv(OUT/(stem+'_view.csv'),index=False);image_table(table,figdir/(stem+'.png'),title)
 source_figs=[]
 for source in m.source.unique():
  fig,axes=plt.subplots(1,3,figsize=(11.2,3.5));true=[0,4] if source.startswith('TON') else [0] if 'Normal' in source else [3] if source=='CIC_SQL' else [2] if source=='CIC_PortScan' else [1]
  for ax,name in zip(axes,POLICIES):
   cm=pd.read_csv(RUN/source/('CM_'+name+'.csv'),index_col=0).to_numpy();shown=cm[true];ax.imshow(np.log1p(shown),cmap='Blues',vmin=0,vmax=np.log1p(1000),aspect='auto');ax.set_xticks(range(5),['Normal','DDoS','Scan','SQL','Ransom'],rotation=30,ha='right');ax.set_yticks(range(len(true)),[['Normal','DDoS','Scan','SQL','Ransom'][i] for i in true]);ax.set_title(LABELS[name],fontsize=9);ax.set_xlabel('Prediksi')
   for i in range(len(true)):
    for j in range(5):ax.text(j,i,str(shown[i,j]),ha='center',va='center',fontsize=9,color='white' if shown[i,j]>100 else '#203344')
  fig.suptitle(source+' | '+('label flow turunan' if source.startswith('TON') else 'Normal dari sumber' if 'Normal' in source else 'label skenario capture'),fontsize=11);fig.tight_layout();stem='CM_'+source;fig.savefig(figdir/(stem+'.png'),bbox_inches='tight',dpi=180);plt.close(fig);source_figs.append(stem)
 # Trace examples preserve actual row IDs and probabilities.
 for source in ['CIC_Normal3','TON_Ransom1']:
  sub=pred.query('source==@source');wide=sub.pivot(index='row_id',columns='policy',values='prediction');truth=sub.drop_duplicates('row_id').set_index('row_id').truth;wide['truth']=truth
  if source=='TON_Ransom1':chosen=wide.query("truth=='Ransomware' and weighted_reference!='Normal' and sql_guard_rf_rescue=='Normal'")
  else:chosen=wide.query("truth=='Normal' and weighted_reference=='Normal' and sql_guard_rf_rescue!='Normal'")
  chosen.head(12).reset_index().to_csv(OUT/('trace_examples_'+source+'.csv'),index=False)
  image_table(chosen.head(8).reset_index(),figdir/('trace_'+source+'.png'),'Contoh perubahan aktual - '+source)
 assets=io.BytesIO()
 include=['primary_policy_metrics_verified.csv','all_policy_metrics_verified.csv','all_predictions_verified.csv','dataset_summaries.csv','TON_label_coverage.csv','rule_changes.csv','false_alarm_predicted_classes.csv']+[stem+'_view.csv' for stem in tables]+['trace_examples_CIC_Normal3.csv','trace_examples_TON_Ransom1.csv']
 with zipfile.ZipFile(assets,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name in include:z.write(OUT/name,name)
  z.write(OUT/'independent_verification.json','independent_verification.json')
  for p in figdir.glob('*.png'):z.write(p,'figures/'+p.name)
 def md(s):return dict(cell_type='markdown',metadata={},source=s)
 def code(s):return dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[])
 cells=[md('# Uji independen CICIoT2023 dan TON_IoT\n\n**Target FPR di bawah 8% belum tercapai.** Model dan aturan tetap; tidak ada training ulang. 9.000 target unik, 27.000 keputusan dari tiga predictor utama. FPR kandidat pada 2.000 Normal CIC adalah 31,9%; pada 2.000 Normal TON yang terpetakan 18,85%. Angka 0,14% sebelumnya berlaku pada blok capture lama.\n\nCIC attack labels adalah label skenario capture. TON memakai label flow penulis dengan interval waktu terkuantisasi; banyak paket tidak terpetakan. Tidak ada satu testbed independen lima kelas yang seragam. Notebook ini meninjau saved results, tanpa model loading atau fitting.'),code("from pathlib import Path\nimport json,base64,io,zipfile\nimport pandas as pd\nfrom IPython.display import display,Image\np=list(Path.cwd().glob('UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb'))\nif not p:p=list(Path.cwd().glob('notebooks/independent_20261109/UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb'))\nassert len(p)==1\nnb=json.loads(p[0].read_text('utf8'))\nassets=zipfile.ZipFile(io.BytesIO(base64.b64decode(nb['metadata']['independent_assets_zip'])))\ndef table(name):return pd.read_csv(assets.open(name),low_memory=False)\ndef picture(name):display(Image(data=assets.read('figures/'+name+'.png')))\nprint(json.loads(assets.read('independent_verification.json'))['status'])"),md('## FPR pada sumber eksternal\n\nTON hanya memakai paket Normal yang terpetakan ke label flow. Pooling di sini deskriptif; dua file tidak membuktikan dua sesi independen.'),code("picture('FPR_independent')\ndisplay(table('dataset_summaries.csv'))")]
 for stem,(_,title) in tables.items():cells.extend([md('## '+title),code("picture('"+stem+"')\ndisplay(table('"+stem+"_view.csv'))")])
 for stem in source_figs:cells.extend([md('## '+stem.replace('_',' ')),code("picture('"+stem+"')")])
 for source in ['CIC_Normal3','TON_Ransom1']:cells.extend([md('## Contoh prediksi aktual - '+source),code("picture('trace_"+source+"')\ndisplay(table('trace_examples_"+source+".csv'))")])
 cells.extend([md('## Menelusuri semua prediksi\n\nFilter `source`, `truth`, `policy`, atau `row_id`. Probabilitas lima kelas tersedia untuk setiap keputusan utama.'),code("predictions=table('all_predictions_verified.csv')\nprint('Keputusan tersimpan:',len(predictions),'| target unik:',predictions.row_id.nunique())\ndisplay(predictions.head(15))\ndisplay(predictions.query(\"source=='TON_Ransom1' and truth=='Ransomware' and policy=='sql_guard_rf_rescue' and prediction=='Normal'\").head(15))"),md('## Makna untuk penelitian\n\nPerbaikan guard bersifat spesifik pada data sebelumnya. Pada data eksternal, guard dapat menekan alarm yang sebenarnya berasal dari trafik berlabel serangan. Recall jenis Ransomware tetap rendah. Model tidak dipilih ulang memakai hasil ini. Paper/supplementary lama belum otomatis diganti. Source, model/hash, fitur, target, probabilitas dan label mapping tersedia dalam paket evidence. Metrik skenario CIC tidak boleh ditulis sebagai recall serangan paket terkonfirmasi; label flow TON juga bukan anotasi paket asli.')])
 for i,c in enumerate(cells):c['id']=f'independent-{i:03}'
 nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.11'},'ta_hafid':{'no_model_deserialization':True},'independent_assets_zip':base64.b64encode(assets.getvalue()).decode()},cells=cells);folder=ROOT/'notebooks/independent_20261109';folder.mkdir(parents=True,exist_ok=True);(folder/'UJI_INDEPENDEN_HASIL_DAN_PREDIKSI_20261109.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf8');print('INDEPENDENT_VIEWS_AND_NOTEBOOK_CREATED',len(cells),flush=True)
if __name__=='__main__':main()
