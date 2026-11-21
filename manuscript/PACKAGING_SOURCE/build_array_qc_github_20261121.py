from pathlib import Path
import importlib.util,json,re,shutil,hashlib
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'paper/array_20261121_quality_control_github'; L=W/'latex'
URL='https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence'
spec=importlib.util.spec_from_file_location('old',ROOT/'scripts/build_array_clean_20261121.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def tex(t):
 t=t.replace('τb','τ_b').replace('ε₆₄ =','epsilon64 =')
 s=old.latex(t)
 s=s.replace('epsilon64',r'\(\epsilon_{64}\)')
 return s
def blocks(d):
 i=-1
 for x in d.element.body:
  if x.tag.endswith('}p'):
   i+=1;yield 'p',i,Paragraph(x,d).text.strip()
  elif x.tag.endswith('}tbl'):yield 't',i,[[c.text for c in r.cells] for r in Table(x,d).rows]
def main():
 L.mkdir(parents=True,exist_ok=True)
 base=ROOT/'paper/array_20261121_gnb_focus/latex'
 for n in ['elsarticle.cls','elsarticle-num.bst','references.bib']:shutil.copy2(base/n,L/n)
 for n in ['figures','figure_sources']:shutil.copytree(base/n,L/n,dirs_exist_ok=True)
 doc=Document(W/'inputs/ARRAY_Hafid_Submission_QUALITY_CONTROL_FINAL_20261121.docx')
 source=(base/'main.tex').read_text('utf8');pre=source[:source.index(r'\begin{document}')]
 front=source[source.index(r'\begin{frontmatter}'):source.index(r'\begin{abstract}')]
 output=[pre,r'\begin{document}',r'\nocite{'+','.join(old.CITEKEYS)+'}',front,r'\begin{abstract}',tex(doc.paragraphs[5].text),r'\end{abstract}',r'\begin{keyword}IoT intrusion detection \sep multiscale endpoint context \sep weighted voting \sep prediction-preserving early exit \sep SHAP auditing \sep reproducibility\end{keyword}\end{frontmatter}']
 cap=None;tn=0;fn=0;changes=[]
 for kind,i,t in blocks(doc):
  if kind=='t':
   tn+=1;output.append(old.make_table(tn,cap,t));cap=None;continue
  if i<7 or not t:continue
  if t=='References':break
  if t.startswith('Edge-IIoTset is distributed'):
   before=t;t=t.replace('as a separate supporting-data package for the journal submission and reviewer-access system','in the public repository '+URL+' (release qc-20261121)')
   changes.append({'paragraph':i,'before':before,'after':t})
  m=re.match(r'Table (\d+):\s*(.*)',t)
  if m:cap=m[2];continue
  m=re.match(r'Figure (\d+)\.\s*(.*)',t)
  if m:
   fn+=1;n=int(m[1]);output.append(r'\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/'+old.FIGURES[n-1]+r'.pdf}\caption{'+tex(m[2])+r'}\label{fig:main'+str(n)+r'}\end{figure}');continue
  m=re.match(r'^\d+\.\d+\.\s*(.*)',t)
  if m:output.append(r'\subsection{'+tex(m[1])+'}');continue
  m=re.match(r'^\d+\.\s*(.*)',t)
  if m and i in [7,21,30,40,69,115,124]:output.append(r'\section{'+tex(m[1])+'}');continue
  if i in [127,129,133]:output.append(r'\section*{'+tex(t)+'}');continue
  if i in [48,52,61,63]:output.append(old.with_equations({48:47,52:51,61:60,63:62}[i],t));continue
  output.append(tex(t))
 output += [r'\bibliographystyle{elsarticle-num}',r'\bibliography{references}',r'\end{document}']
 assert tn==7 and fn==5
 (L/'main.tex').write_text('\n\n'.join(output),encoding='utf8')
 # Editable tables and complete revised supervisor prose; no PDF-to-image manuscript.
 sd=Document(W/'inputs/ARRAY_Hafid_Supplementary_QUALITY_CONTROL_FINAL_20261121.docx')
 so=[r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=17mm]{geometry}\usepackage{booktabs,array,amsmath,xurl,float}
\usepackage[hidelinks]{hyperref}\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\setlength{\emergencystretch}{2em}
\renewcommand{\thetable}{S\arabic{table}}\begin{document}
\begin{center}\large\bfseries Supplementary material\par\normalsize Multiscale endpoint context and prediction-preserving execution\\for explainable IoT intrusion detection\par\small M Hafid Ramadhan and Satria Mandala\end{center}''']
 sn=0
 for kind,i,t in blocks(sd):
  if kind=='t':
   sn+=1;n=len(t[0]);widths={1:[.16,.14,.14,.14,.14,.14,.14],2:[.17,.19,.125,.125,.125,.125,.14],3:[.35,.21,.22,.22],4:[.30,.46,.24],5:[.31,.12,.20,.185,.185]}[sn]
   fmt=''.join('L{'+str(w)+r'\dimexpr\textwidth-'+str(2*n)+r'\tabcolsep\relax}' for w in widths)
   so += [r'\begin{table}[H]\centering\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}',r'\caption{'+tex(cap)+r'}\label{sup:'+list(old.SUP.values())[sn-1]+'}',r'\begin{tabular}{'+fmt+r'}\toprule',' & '.join(tex(c) for c in t[0])+r'\\\midrule']
   so += [' & '.join(tex(c) for c in row)+r'\\' for row in t[1:]]
   so += [r'\bottomrule\end{tabular}\end{table}'];continue
  if i<3 or not t:continue
  m=re.match(r'Table S\d+\.\s*(.*)',t)
  if m:cap=m[1];continue
  if re.match(r'S\d+\.',t) or t.startswith('Reproducibility'):so.append(r'\subsection*{'+tex(t)+'}');continue
  if t.startswith('The R1–R4 archive is organized'):
   before=t;t=t.replace('as a separate supporting-data package for upload with the manuscript through the journal submission system','in the public repository '+URL+' (release qc-20261121)');changes.append({'supplement_paragraph':i,'before':before,'after':t})
  so.append(tex(t))
 so.append(r'\end{document}');st='\n\n'.join(so)
 for s,k in old.SUP.items():st=st.replace(r'\ref*{sup:'+k+'}',s)
 (L/'SUPPLEMENTARY').mkdir(exist_ok=True);(L/'SUPPLEMENTARY/supplement.tex').write_text(st,encoding='utf8')
 (W/'editorial_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf8')
 print('Built main:7 tables/5 figures; supplement:5 tables; preserved latest supervisor wording; linked public repository')
if __name__=='__main__':main()
