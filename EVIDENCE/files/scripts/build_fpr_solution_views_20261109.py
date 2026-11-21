"""Tables/figures and an embedded-data review notebook from verified saved data."""
from pathlib import Path
import base64,io,json,zipfile,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/FPR_SOLUTION_20261109';KEY='sql_guard_rf_rescue'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})

def image_table(df,path,title):
    text=df.copy()
    for c in text:
        if pd.api.types.is_float_dtype(text[c]):text[c]=text[c].map(lambda x:'' if pd.isna(x) else f'{x:.4f}')
    width=max(12,min(24,int(100/len(text.columns))))
    wrap=lambda s:'\n'.join(textwrap.wrap(str(s).replace('_',' '),width=width,break_long_words=True))
    values=[[wrap(v) for v in row] for row in text.astype(str).values]
    fig,ax=plt.subplots(figsize=(11.7,max(2.7,.46*len(df)+1.7)));ax.axis('off');ax.set_title(title,loc='left',fontsize=13,pad=16)
    table=ax.table(cellText=values,colLabels=[wrap(c) for c in text.columns],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8.2);table.scale(1,1.8)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#dce2e8')
        if i==0:cell.set_facecolor('#173c5a');cell.set_text_props(color='white',weight='bold');cell.set_height(cell.get_height()*2.15)
        else:
            if i%2==0:cell.set_facecolor('#f2f5f8')
            if '\n' in cell.get_text().get_text():cell.set_height(cell.get_height()*1.5)
    fig.savefig(path,bbox_inches='tight');plt.close(fig)

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    (OUT/'figures').mkdir(exist_ok=True);m=pd.read_csv(OUT/'metrics_all_cached.csv');b=pd.read_csv(OUT/'metrics_later_blocks.csv');g=pd.read_csv(OUT/'independent_safety_gates.csv');d=pd.read_csv(OUT/'predictions_all.csv',low_memory=False);p=pd.read_csv(OUT/'metrics_per_capture.csv');cause=pd.read_csv(OUT/'expert_disagreement_cause.csv')
    fpr=[]
    for name in ['weighted_reference','random_forest',KEY]:
        a=m.query("cohort=='additional760' and policy==@name").iloc[0];fpr.append(dict(cohort='Lama 760',policy=name,n=int(a.n),FP=int(a.false_alarms),FPR_percent=a.FPR_percent))
        for _,row in b.query('policy==@name').iterrows():fpr.append(dict(cohort='Blok '+str(int(row.block)),policy=name,n=int(row.n),FP=int(row.false_alarms),FPR_percent=row.FPR_percent))
    fpr=pd.DataFrame(fpr);fpr.to_csv(OUT/'FPR_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(10.4,3.8));x=np.arange(4)
    for name,dx,color,label in [('weighted_reference',-.18,'#c76a40','Weighted voting lama'),(KEY,.18,'#16757b','SQL guard + RF rescue')]:
        q=fpr.query('policy==@name');ax.bar(x+dx,q.FPR_percent,.34,color=color,label=label)
        for xx,(_,r) in zip(x+dx,q.iterrows()):ax.text(xx,r.FPR_percent+.9,f'{int(r.FP)}/{int(r.n)}\n{r.FPR_percent:.2f}%',ha='center',va='bottom',fontsize=8.5)
    ax.axhline(8,color='#a51c30',ls='--',lw=1,label='Target 8%');ax.set_xticks(x,['760 lama','Blok 1: 769','Blok 2: 699','Blok 3: 697']);ax.set_ylim(0,67);ax.set_ylabel('False-positive rate (%)');ax.legend(loc='upper left',bbox_to_anchor=(0,1.23),ncol=3,frameon=False);fig.tight_layout();fig.savefig(OUT/'figures/FPR_comparison.png',bbox_inches='tight');plt.close(fig)
    names=['Normal','DDoS','Scan','SQL','Ransom'];fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
    for ax,name,title in zip(axs,['weighted_reference',KEY],['Voting lama - V seed42','SQL guard + RF rescue - V seed42']):
        cm=pd.read_csv(OUT/('CM_V_seed42__'+name+'.csv'),index_col=0).to_numpy();im=ax.imshow(np.log1p(cm),cmap='Blues');ax.set_xticks(range(5),names,rotation=30,ha='right');ax.set_yticks(range(5),names);ax.set_title(title,fontsize=11);ax.set_xlabel('Prediksi');ax.set_ylabel('Kelas sebenarnya')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if np.log1p(cm[i,j])>5 else '#23364a',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'figures/CM_V_seed42.png',bbox_inches='tight');plt.close(fig)
    summary=pd.read_csv(ROOT/'runs/FPR_SQL_GUARD_20261109_01/efficiency_summary.csv');fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    for ax,co,title in zip(axs,['V1000','later'],['V: batch 1.000','Normal blok 3: batch 697']):
        q=summary.query('cohort==@co').set_index('policy').loc[['random_forest','weighted_reference',KEY]];ax.bar(['RF','Voting lama','Aturan baru'],q['median'],color=['#7c8e9c','#c76a40','#16757b']);ax.set_title(title);ax.set_ylabel('Median inferensi (ms/batch)');ax.tick_params(axis='x',rotation=12)
        for j,v in enumerate(q['median']):ax.text(j,v+1,f'{v:.2f}',ha='center')
        ax.set_ylim(0,q['median'].max()*1.22)
    fig.tight_layout();fig.savefig(OUT/'figures/latency_comparison.png',bbox_inches='tight');plt.close(fig)
    # Focused tables for notebook output images.
    tables={
        'FPR_summary':(fpr,'FPR - angka aktual, bukan simulasi'),
        'attack_validation':(m.query("cohort=='V_seed42'")[['policy','FPR_percent','attack_to_Normal','macro_F1_percent','balanced_accuracy_percent','DDoS_F2_percent']],'Trade-off pada V seed42 (10.358 target)'),
        'safety_gates':(g[['cohort','passed','macro_F1_delta_pp','attack_to_Normal_delta']],'Tujuh pemeriksaan recall serangan pada cohort historis'),
        'per_class_recall':(m.query("cohort=='V_seed42'")[['policy']+['recall_'+c for c in ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']]],'Recall per kelas V seed42 (%)'),
        'per_capture_block3':(p.query('block==3 and policy==@KEY')[['source','planned','n','false_alarms','FPR_percent']],'Normal blok ketiga - distribusi per capture'),
        'expert_diagnostic':(cause[['model','weight','false_alarms','SQL_predictions','weighted_mean_SQL_minus_Normal_on_321_WV_false_alarms']].rename(columns={'weighted_mean_SQL_minus_Normal_on_321_WV_false_alarms':'mean weighted SQL-Normal margin'}),'Diagnostic 760 Normal - ketidaksepakatan antar-model'),
        'changed_attacks':(d.query("cohort=='V_seed42' and changed and true_class!='Normal'")[['row_id','true_class','weighted_prediction','RF_prediction','candidate_prediction']],'Dua serangan terangkat menjadi alarm; jenis masih salah'),
        'latency_paired':(pd.read_csv(OUT/'paired_latency_comparison.csv')[['cohort','policy','median_ms','mean_paired_delta_ms','bootstrap_95_low_ms','bootstrap_95_high_ms']],'Latensi berpasangan: kandidat minus voting (ms)'),
        'memory':(summary[['cohort','policy','RSS_delta_mean_MiB','RSS_delta_SD_MiB']],'Delta peak RSS inferensi; tiga cold process, bukan RAM total')}
    for stem,(df,title) in tables.items():
        df.to_csv(OUT/(stem+'_view.csv'),index=False);image_table(df,OUT/'figures'/(stem+'.png'),title)
    assets=io.BytesIO()
    with zipfile.ZipFile(assets,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(OUT.glob('*.csv')):z.write(path,path.name)
        z.write(OUT/'verification.json','verification.json')
        for path in sorted((OUT/'figures').glob('*.png')):z.write(path,'figures/'+path.name)
    def md(text):return dict(cell_type='markdown',metadata={},source=text)
    def code(text):return dict(cell_type='code',metadata={},source=text,execution_count=None,outputs=[])
    cells=[md('# Solusi FPR di bawah 8% - hasil 9 November 2026\n\nAturan baru: SQL guard + RF rescue. Ini predictor yang diubah menggunakan expert yang telah fitted, tanpa training baru. Hasil berasal dari artefak lokal yang telah diverifikasi. Notebook ini hanya meninjau saved predictions, tidak memuat model.\n\nBlok 1/2 dan tujuh cohort serangan dipakai secara retrospektif. Aturan dibekukan sebelum blok 3 (Normal saja, capture lama). Tidak ada test serangan independen. Dua Scan yang tadinya Normal menjadi Ransomware: alarm terangkat, jenis tetap salah. PDF jurnal sebelumnya belum diubah oleh eksperimen ini.'),code("from pathlib import Path\nimport json,base64,io,zipfile\nimport pandas as pd\nfrom IPython.display import display,Image\n_candidates=list(Path.cwd().glob('FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb'))\nif not _candidates:\n    _candidates=list(Path.cwd().glob('notebooks/fpr_solution_20261109/FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb'))\nassert len(_candidates)==1, 'Jalankan dari folder notebook atau root proyek'\n_nb=json.loads(_candidates[0].read_text('utf8'))\n_assets=zipfile.ZipFile(io.BytesIO(base64.b64decode(_nb['metadata']['fpr_evidence_zip'])))\ndef table(name):return pd.read_csv(_assets.open(name),low_memory=False)\ndef picture(name):display(Image(data=_assets.read('figures/'+name+'.png')))\nprint(json.loads(_assets.read('verification.json'))['status'])")]
    captions={'FPR_summary':'760 lama 42,24% menjadi0%. Blok ketiga396/697 menjadi1/697=0,14%;3unscorable => worst-case4/700=0,57%. Blok baru Normal saja tidak mengukur sensitivitas serangan.','attack_validation':'V Normal FPR0,9326% menjadi1,0104% (+3FP). Macro-F1turun0,02977pp; binary attack-to-Normal2menjadi0. Ini bukan peningkatan klasifikasi menyeluruh.','safety_gates':'Semua7cohort memenuhi tidak ada penurunan recall kelas serangan, attack-to-Normal tidak meningkat, macro-F1drop<=0,1pp (batas operasional). Data telah diamati; bukan uji non-inferiority independen.','expert_diagnostic':'Seluruh321alarm pada760Normal adalahSQL. Margin menunjukkan kontribusi agregasi, bukan bukti kausal sumber capture atau isi payload.','changed_attacks':'Periksa y_true dan jenis prediksi. Mengubah Normal menjadi Ransomware pada trueScan tidak berarti klasifikasi jenis berhasil.'}
    for stem,(df,title) in tables.items():cells.extend([md('## '+title+'\n\n'+captions.get(stem,'')),code("picture('"+stem+"')\ndisplay(table('"+stem+"_view.csv'))")])
    for stem,title in [('FPR_comparison','Perbandingan false-positive rate'),('CM_V_seed42','Confusion matrix lima kelas'),('latency_comparison','Perbandingan median inferensi')]:cells.extend([md('## '+title),code("picture('"+stem+"')")])
    cells.extend([md('## Cari prediksi aktual\n\n`predictions` memuat seluruh saved decisions yang direkonstruksi; gunakan filter kelas/cohort. `changed` berarti label kandidat berbeda dari voting lama. Jangan mencampur seed/protokol sebagai sampel independen.'),code("predictions=table('predictions_all.csv')\nprint('Baris saved decisions:',len(predictions))\ndisplay(predictions.query(\"cohort=='later_block3' and candidate_prediction!='Normal'\"))\ndisplay(predictions.query(\"cohort=='V_seed42' and changed\"))"),md('## Batas dan penggunaan\n\nAturan dapat menekan trueSQL menjadiNormal jikaRFsalah; risiko ini belum diuji pada capture serangan baru. Semua model/feature extractor lama tetap; benchmark hanya prepared-feature inference. Untuk inferensi ulang gunakan scriptCLI di paket evidence setelah hash dan source cocok. Jangan menyalin angka41,75% replay lama ke predictor baru. Hasil ini layak dibawa ke dospem sebagai kandidat perbaikan, bukan klaim produksi atau acceptance jurnal.')])
    for i,cell in enumerate(cells):cell['id']=f'fpr-{i:03d}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'ta_hafid':{'no_model_deserialization':True},'fpr_evidence_zip':base64.b64encode(assets.getvalue()).decode()},cells=cells)
    dest=ROOT/'notebooks/fpr_solution_20261109';dest.mkdir(parents=True,exist_ok=True);(dest/'FPR_SOLUSI_DAN_PREDIKSI_20261109.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    print('VIEWS_AND_NOTEBOOK_CREATED',len(tables),'tables',len(cells),'cells')
if __name__=='__main__':main()
