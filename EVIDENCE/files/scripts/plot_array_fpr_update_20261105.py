"""Vector flow diagram of the explicitly changed selective predictor."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Polygon,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/array_20261105_fpr_update/elsarticle/figures/fpr_reconciliation.pdf'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9,4.2));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',linewidth=1.2,edgecolor='#385d78',facecolor=color));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def diamond(x,y,w,h,text):
    ax.add_patch(Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)],closed=True,facecolor='#edf3f8',edgecolor='#385d78',linewidth=1.2));ax.text(x,y,text,ha='center',va='center',fontsize=10.5)
def arrow(a,b,label=None,at=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#385d78',linewidth=1.3))
    if label:ax.text(*(at or ((a[0]+b[0])/2,(a[1]+b[1])/2)),label,fontsize=10,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
box(.08,.83,.84,.12,'Compute original five-expert voting f(x); retain RF posterior r(x)','#edf3f8')
diamond(.30,.59,.40,.32,'Voting: SQL; RF: Normal\nRF Normal probability\nat least 0.5?')
diamond(.75,.59,.40,.32,'Usable content gives\nSQL evidence or tie?')
box(.10,.10,.39,.16,'Keep original full\nvoting posterior f(x)','#edf3f8')
box(.62,.10,.31,.16,'Use complete\nRF posterior r(x)','#e6f1eb')
arrow((.30,.83),(.30,.76))
arrow((.505,.59),(.545,.59),'Yes',(.525,.65))
arrow((.26,.45),(.26,.275),'No',(.22,.36))
arrow((.69,.46),(.43,.275),'Yes',(.57,.36))
arrow((.80,.45),(.80,.275),'No',(.85,.36))
ax.text(.5,.025,'Post-observation candidate; no attack activated this rule in G/C/V.',ha='center',va='bottom',fontsize=10,color='#445b6e')
fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
fig.savefig(OUT,bbox_inches='tight');plt.close(fig);print(OUT)
