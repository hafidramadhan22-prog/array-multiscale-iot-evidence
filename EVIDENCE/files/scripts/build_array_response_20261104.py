"""Create a supervisor response PDF from the explicit six-point response."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/array_20261104_final_review/response';OUT.mkdir(exist_ok=True)
def esc(t):
    t=t.replace('–','--').replace('—','---').replace('×',r'$\times$').replace('≥',r'$\geq$').replace('≤',r'$\leq$').replace('−','-')
    urls={}
    def hold(m):
        key='URLTOKEN'+str(len(urls));urls[key]=r'\url{'+m[0]+'}';return key
    t=re.sub(r'https?://[^\s;)]+',hold,t)
    def holdpath(m):
        key='URLTOKEN'+str(len(urls));urls[key]=r'\path{'+m[0]+'}';return key
    t=re.sub(r'\b(?:output|reports|runs|paper|docs)/[^\s,;]+',holdpath,t)
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:t=t.replace(a,b)
    t=re.sub(r'\*\*([^*]+)\*\*',r'\\textbf{\1}',t)
    for key,v in urls.items():t=t.replace(key,v)
    # Long machine-readable identifiers should be breakable.
    t=re.sub(r'(?<![A-Za-z])[a-f0-9]{64}(?![A-Za-z0-9])',lambda m:r'\path{'+m[0]+'}',t)
    return t

lines=(ROOT/'docs/ARRAY_FINAL_REVIEW_20261104.md').read_text(encoding='utf8').splitlines()
body=[];i=0
while i<len(lines):
    line=lines[i]
    if line.startswith('|'):
        group=[]
        while i<len(lines) and lines[i].startswith('|'):group.append([c.strip() for c in lines[i].strip('|').split('|')]);i+=1
        group=[r for r in group if not all(re.fullmatch(r'[:\- ]+',c) for c in r)]
        n=len(group[0]);widths=[.25,.4,.35] if n==3 else [1/n]*n
        cols=''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+'\\dimexpr\\linewidth-'+str(2*n)+r'\tabcolsep\relax}' for w in widths)
        body.append(r'\small\begin{longtable}{'+cols+r'}\toprule')
        body.append(' & '.join(r'\textbf{'+esc(c)+'}' for c in group[0])+r'\\\midrule\endhead')
        for row in group[1:]:body.append(' & '.join(esc(c) for c in row)+r'\\\addlinespace[4pt]')
        body.append(r'\bottomrule\end{longtable}\normalsize');continue
    if line.startswith('# '):body.append(r'\begin{center}\Large\bfseries '+esc(line[2:])+r'\end{center}')
    elif line.startswith('## '):body.append(r'\section*{'+esc(line[3:])+'}')
    else:body.append(esc(line))
    i+=1
text=r'''\documentclass[12pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=23mm]{geometry}
\usepackage{array,booktabs,longtable,xurl,amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\lhead{Array -- tanggapan revisi}\rhead{4 November 2026}\cfoot{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{4em}\raggedbottom
\begin{document}
\renewcommand{\normalsize}{\fontsize{11}{13.2}\selectfont}\normalsize
'''+ '\n'.join(body)+ '\n'+r'\end{document}'
(OUT/'response.tex').write_text(text,encoding='utf8')
print(OUT/'response.tex')
