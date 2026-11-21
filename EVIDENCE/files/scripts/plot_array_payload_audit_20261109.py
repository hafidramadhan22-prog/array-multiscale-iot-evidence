"""Programmatic figure from two checksum-audited actual training records."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'paper/array_20261109_payload_audit/elsarticle/figures'
rows=[json.loads(s) for s in (ROOT/'reports/PAYLOAD_EVIDENCE_20261105_01/actual_payload_traces.jsonl').read_text('utf8').splitlines()]
normal=next(r for r in rows if r['row_id']==105044);sql=next(r for r in rows if r['row_id']==23740)
assert normal['extracted_content']=='24.68 76.42'
assert sql['extracted_content']=='2 Submitid=53 ORDER BY 1-- MEbL'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(10,6.5));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')

def box(x,y,w,h,title,text,color,mono=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor=color,edgecolor='#486476',linewidth=1))
    ax.text(x+.017,y+h-.025,title,ha='left',va='top',fontsize=13.5,fontweight='bold',color='#153b50')
    ax.text(x+.017,y+h-.075,text,ha='left',va='top',fontsize=12.4,fontfamily='DejaVu Sans Mono' if mono else 'DejaVu Sans',color='#20394a',linespacing=1.5)

for x,r,title in [(.025,normal,'Normal: canonical row 105044'),(.525,sql,'SQL: canonical row 23740')]:
    ax.text(x,.965,title,fontsize=15.2,fontweight='bold',color='#153b50',va='top')
    prefix=' '.join(r['tcp_payload_hex'][i:i+2] for i in range(0,24,2))
    box(x,.72,.45,.17,'Stored TCP payload (hex prefix)',prefix+'\n... (full bytes retained)','#edf3f7',True)
    text='MQTT topic and measurement\nTemperature_and_Humidity' if r is normal else 'HTTP GET request\n/DVWA/vulnerabilities/sqli_blind/'
    box(x,.45,.45,.18,'Decoded current packet',text,'#edf3f7',False)
    content=r['extracted_content']
    if r is sql:content='2 Submitid=53\nORDER BY 1-- MEbL'
    box(x,.18,.45,.18,'Content passed to classifier',content,'#e4f0eb',True)
    for y0,y1 in [(.71,.64),(.44,.37)]:ax.add_patch(FancyArrowPatch((x+.225,y0),(x+.225,y1),arrowstyle='-|>',mutation_scale=16,color='#486476',linewidth=1.2))
ax.text(.5,.07,'The two content types do not form a matched benign/malicious HTTP pair.',ha='center',va='center',fontsize=13,color='#435e70')
ax.text(.5,.025,'Headers are excluded from the extracted HTTP content. No packet or request is executed.',ha='center',va='center',fontsize=12,color='#435e70')
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
fig.savefig(DEST/'payload_trace_audit.pdf',bbox_inches='tight')
fig.savefig(DEST/'payload_trace_audit.png',dpi=190,bbox_inches='tight')
plt.close(fig)
print('ACTUAL_RECORD_FIGURE_COMPLETE')
