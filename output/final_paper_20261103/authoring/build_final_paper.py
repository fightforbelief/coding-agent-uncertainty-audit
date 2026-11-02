"""Build a full manuscript from the completed, hash-locked P8 analyses.

Reads saved results only; never refits a model or creates new test predictions.
The PDF, Markdown and LaTeX share the same narrative and generated table rows.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'analysis_v2/original_plan/v3_full_analysis_20261103'
MEAS=ROOT/'analysis_v2/original_plan/owner_authorized_ai_adjudication_20261103'
OUT=ROOT/'output/final_paper_20261103'
PDF=ROOT/'output/pdf/P8_final_paper_20261103.pdf'
WIDTH=A4[0]-108


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def num(v,d=6):return '--' if pd.isna(v) else f'{v:.{d}f}'
def effect(v):
    if pd.isna(v):return '--'
    return f'{v:.3g}' if (v!=0 and (abs(v)<.001 or abs(v)>1000)) else num(v,4)
def ci(r,d=6):return f'[{num(r.delta_ap_low,d)}, {num(r.delta_ap_high,d)}]'


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def datasets():
    assert read(RUN/'WORKFLOW_STATUS.json')['stage']=='COMPLETE'
    for name,digest in read(RUN/'WORKFLOW_STATUS.json')['files'].items():
        assert sha(RUN/'publication'/name)==digest,name
    for name,digest in read(MEAS/'AI_FINAL_LABEL_LOCK.json')['files'].items():assert sha(MEAS/name)==digest,name
    metrics=pd.read_csv(RUN/'checkpoints/checkpoint_metrics.csv')
    assert len(metrics)==34 and metrics.primary_exploratory_endpoint.sum()==1
    for p in (RUN/'checkpoints').glob('*/metrics.json'):
        m=read(p);assert sha(p.parent/'predictions.parquet')==m['prediction_sha256']
        for name,r in m['models'].items():
            if name!='M0':assert r['bootstrap_valid']==1000 and not r['bootstrap_skipped']
    return metrics


def tables(metrics):
    meta=pd.read_parquet(ROOT/'analysis_v2/meta.parquet')
    exclusions=set((ROOT/'analysis_v2/audit/audit_ids_EXCLUDE.txt').read_text().splitlines())
    eligible=meta[meta.resolved.isin([0,1])&~meta.trajectory_id.isin(exclusions)]
    assert len(eligible)==159867 and eligible.repo.nunique()==2349 and eligible.issue_id.nunique()==17086
    t={}
    def add(key,caption,headers,rows,widths=None):t[key]=dict(caption=caption,headers=headers,rows=rows,widths=widths)
    rows=[]
    for split in ['train','validation','test']:
        p=eligible[eligible.split.eq(split)]
        rows.append([split.title(),f'{len(p):,}',f'{p.repo.nunique():,}',f'{p.issue_id.nunique():,}',num(p.resolved.eq(0).mean())])
    add('TABLE_SPLIT','Table 1. Known-label modeling pool after audit exclusions, before checkpoint-reaching filters. Positive outcome is task failure.',
        ['Partition','Trajectories','Repositories','Issues','Failure share'],rows,[.19,.21,.2,.18,.22])
    m=pd.read_csv(MEAS/'final_ai_measurement.csv');m=m[m.scaffold.eq('all')]
    names={'R5_detector_positive_frame':'R5 positive (A)','R15_detector_positive_frame':'R15 positive (A+B)',
        'R15_detector_known_negative_frame':'R15 known negative (C)','R15_missing_command_diagnostic':'R15 missing command (D)',
        'R5_ambiguous_command_diagnostic':'R5 ambiguous (AMB5)','R15_ambiguous_command_diagnostic':'R15 ambiguous (AMB15)'}
    rows=[[names[r.frame],int(r.sample_memberships),f'{100*r.positive_share_all:.2f}%',f'{100*r.uncertain_share:.2f}%',f'{100*r.positive_share_determinate:.2f}%'] for r in m.itertuples(index=False)]
    add('TABLE_MEASUREMENT','Table 2. Stratum-weighted AI-reconciled command labels within the declared audit frames. Positive/determinate excludes uncertain labels; it is not gold-standard precision. Sample n counts memberships in the indicated summary.',
        ['Audit frame','Sample n','Positive / all','Uncertain / all','Positive / determinate'],rows,[.32,.11,.18,.18,.21])
    main=metrics[(metrics.kind=='logistic')&(metrics.population=='main')]
    rows=[[r.model,num(r.ap),num(r.auroc),num(r.brier)] for r in main[main.k.eq(15)].itertuples(index=False)]
    add('TABLE_PRIMARY','Table 3. Primary-checkpoint logistic results (K15; test n = 20,012). Higher AP/AUROC and lower Brier are better; no operational utility threshold was tested.',
        ['Model','Average precision','AUROC','Brier'],rows)
    rows=[]
    for k in [5,10,15,20,30]:
        p=main[main.k.eq(k)].set_index('model');r=p.loc['M2']
        rows.append([k,f'{int(r.n_test):,}',num(p.loc['M0','ap']),num(p.loc['M1','delta_ap']),num(r.delta_ap),ci(r)])
    add('TABLE_CHECKPOINTS','Table 4. All planned main-population logistic contrasts. Intervals are paired 95% repository-bootstrap percentiles for M2 minus M0. K15 is the preselected primary exploratory contrast.',
        ['K','Test n','M0 AP','M1 Delta AP','M2 Delta AP','M2 95% interval'],rows,[.055,.115,.15,.17,.17,.34])
    rows=[[int(r.k),r.model,num(r.ap),num(r.delta_ap),ci(r) if r.model=='M2' else '--'] for r in metrics[metrics.kind.eq('hist_gradient_boosting')].itertuples(index=False)]
    add('TABLE_NONLINEAR','Table 5. Prespecified secondary nonlinear baseline. The K15 M2 contrast is positive, but does not replace the primary logistic endpoint.',
        ['K','Model','AP','Delta AP vs M0','95% paired interval'],rows,[.065,.11,.19,.22,.415])
    odds=pd.read_csv(RUN/'within_issue/within_issue_odds_ratios.csv')
    fits={}
    stability=[]
    for p in (RUN/'within_issue').glob('*_k*.json'):
        f=read(p);assert f['status']=='COMPLETE'
        assert f['bootstrap_valid']+len(f['bootstrap_failed'])==500
        fits[(f['analysis_name'],f['k'])]=f
        stability.append([f['analysis_name'],f['k'],f['n_rows'],f['bootstrap_valid'],len(f['bootstrap_failed'])])
    assert len(fits)==14 and sum(r[4] for r in stability)==860
    def or_interval(r):return f'[{effect(r.or_low)}, {effect(r.or_high)}]'
    rows=[]
    for k in [10,15]:
        for name in ['A','B','C']:
            f=fits[(name,k)];r=odds[(odds.analysis==name)&(odds.k==k)&(odds.feature=='ran_tests')].iloc[0]
            rows.append([k,name,f'{f["n_rows"]:,}','--' if f['n_groups'] is None else f'{f["n_groups"]:,}',num(r.odds_ratio,4),or_interval(r),f'{f["bootstrap_valid"]}/500'])
    add('TABLE_WITHIN','Table 6. Main command-presence associations with failure. A: ordinary logistic; B: issue conditional; C: issue/configuration conditional. Intervals use successful refits only; coverage is not established when refit failure is substantial.',
        ['K','Model','Rows','Groups','Failure OR','95% interval','Valid refits'],rows,[.055,.075,.14,.14,.15,.265,.175])
    # These are descriptive counts of the already frozen feature matrix.
    feature_path=ROOT/'analysis_v2/original_plan/v3_features_20261103/features_v3.parquet'
    assert sha(feature_path)==read(RUN/'checkpoints/MODEL_RUN_CONTEXT.json')['features_sha256']
    features=pd.read_parquet(feature_path)
    features=features[features.resolved.isin([0,1])&~features.trajectory_id.isin(exclusions)]
    rows=[]
    for k in [5,10,15,20,30]:
        p=features[features[f'k{k}_reached'].eq(1)]
        rows.append([k,f'{len(p):,}']+[f'{int(p[f"k{k}_{c}"].sum()):,}' for c in ['ran_tests','test_fail','fail_then_pass']])
    add('TABLE_EVENT_SUPPORT','Table A1. Support for frozen detector proxies among checkpoint-reaching eligible trajectories across all prediction partitions.',
        ['K','Eligible n','Command present','Explicit failure','Failure-then-pass'],rows,[.06,.19,.25,.23,.27])
    rows=[]
    for r in main.itertuples(index=False):
        rows.append([r.k,r.model,num(r.ap),num(r.auroc),num(r.brier),num(r.delta_ap),ci(r) if r.model!='M0' else '--'])
    add('TABLE_ALL_LOGISTIC','Table A2. Complete main-population logistic metrics. All AP contrasts use the same K-specific M0. Intervals are unadjusted exploratory paired-bootstrap summaries.',
        ['K','Model','AP','AUROC','Brier','Delta AP','95% paired interval'],rows,[.05,.08,.14,.14,.14,.14,.31])
    rows=[[r.k,f'{r.n_test:,}',num(r.ap),num(r.delta_ap),ci(r)] for r in metrics[(metrics.kind=='logistic')&(metrics.population=='fixed30')&(metrics.model=='M2')].itertuples(index=False)]
    add('TABLE_FIXED','Table A3. Fixed reached-30 population, M2 versus M0 at each checkpoint.',
        ['K','Test n','M2 AP','Delta AP','95% paired interval'],rows,[.06,.16,.2,.2,.38])
    label=lambda s:s.replace('B_openhands_minimax_m25','B: OH/M').replace('B_openhands_qwen35_122b','B: OH/Q').replace('B_sweagent_minimax_m25','B: SWE/M').replace('B_sweagent_qwen35_122b','B: SWE/Q')
    rows=[[label(name),k,f'{n:,}',valid,failed] for name,k,n,valid,failed in sorted(stability,key=lambda r:(r[1],r[0]))]
    add('TABLE_STABILITY','Table A4. Complete bootstrap refit accounting (500 requested per analysis). A completed point estimate does not imply all resamples were estimable.',
        ['Analysis','K','Point-estimate rows','Valid refits','Failed refits'],rows,[.25,.07,.25,.215,.215])
    rows=[]
    sub=odds[odds.analysis.str.startswith('B_')&odds.feature.eq('ran_tests')].sort_values(['k','analysis'])
    for r in sub.itertuples(index=False):rows.append([label(r.analysis),r.k,effect(r.odds_ratio),or_interval(r),f'{r.bootstrap_valid}/500'])
    add('TABLE_CONFIG','Table A5. Configuration-specific issue-conditional command-presence odds ratios for task failure. Intervals use successful refits only.',
        ['Configuration','K','Failure OR','95% interval','Valid refits'],rows,[.25,.06,.18,.32,.19])
    return t


def inline_html(s):
    # Protect links/code before adding simple emphasis. Never evaluate source HTML.
    s=escape(s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:'<link href="'+m[2]+'" color="#234e69">'+m[1]+'</link>',s)
    s=re.sub(r'`([^`]+)`',lambda m:'<font name="Courier" size="8.2">'+m[1]+'</font>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    return s


def latex_escape(s):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','<':r'\textless{}','>':r'\textgreater{}'}
    return ''.join(replacements.get(c,c) for c in s)


def inline_tex(s):
    # Build from original text rather than escaping a generated command twice.
    parts=[];pos=0
    pattern=r'\[([^\]]+)\]\((https?://[^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*'
    for m in re.finditer(pattern,s):
        parts.append(latex_escape(s[pos:m.start()]))
        if m[1]:parts.append(r'\href{'+latex_escape(m[2])+'}{'+latex_escape(m[1])+'}')
        elif m[3]:
            value=m[3]
            parts.append((r'\nolinkurl{'+value+'}') if len(value)>32 else r'\texttt{'+latex_escape(value)+'}')
        else:parts.append(r'\textbf{'+latex_escape(m[4])+'}')
        pos=m.end()
    parts.append(latex_escape(s[pos:]));return ''.join(parts)


def render(template,tables,figures):
    styles={
        'title':ParagraphStyle('Title',fontName='Times-Bold',fontSize=19,leading=22.5,alignment=TA_CENTER,spaceAfter=10),
        'author':ParagraphStyle('Author',fontName='Times-Roman',fontSize=11,leading=15,alignment=TA_CENTER,spaceAfter=17),
        'h1':ParagraphStyle('H1',fontName='Times-Bold',fontSize=12.2,leading=15,spaceBefore=14,spaceAfter=7,keepWithNext=True),
        'h2':ParagraphStyle('H2',fontName='Times-Bold',fontSize=10.8,leading=13.5,spaceBefore=10,spaceAfter=5,keepWithNext=True),
        'body':ParagraphStyle('Body',fontName='Times-Roman',fontSize=10.2,leading=13.6,alignment=TA_JUSTIFY,spaceAfter=7),
        'small':ParagraphStyle('Small',fontName='Times-Roman',fontSize=8.6,leading=11,spaceAfter=6),
        'caption':ParagraphStyle('Caption',fontName='Times-Roman',fontSize=8.6,leading=11,spaceBefore=5,spaceAfter=8),
        'cell':ParagraphStyle('Cell',fontName='Times-Roman',fontSize=8.1,leading=10),
        'head':ParagraphStyle('Head',fontName='Helvetica-Bold',fontSize=7.5,leading=9.4),
    }
    story=[];md=[];tex=[];in_refs=False
    chunks=template.strip().split('\n\n')
    title=chunks.pop(0).removeprefix('# ');author=chunks.pop(0)
    story.extend([Paragraph(inline_html(title),styles['title']),Paragraph(inline_html(author),styles['author'])])
    md.extend(['# '+title,author])
    tex.extend([r'\documentclass[10pt,a4paper]{article}',r'\usepackage[margin=22mm]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{mathptmx,microtype,graphicx,longtable,booktabs,array}',r'\usepackage[hidelinks,breaklinks]{hyperref}',r'\usepackage{url}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{5pt}',r'\setcounter{secnumdepth}{0}',r'\title{'+latex_escape(title)+'}',r'\author{Yunjie Guo \and Hanqing Yang}',r'\date{3 November 2026}',r'\begin{document}',r'\maketitle'])
    for chunk in chunks:
        token=re.fullmatch(r'\{\{([A-Z_]+)\}\}',chunk)
        if token and token[1] in tables:
            t=tables[token[1]];widths=t['widths'] or [1/len(t['headers'])]*len(t['headers'])
            padding=3 if t['caption'].startswith('Table A') else 5
            widths=np.asarray(widths,dtype=float);widths=widths/widths.sum()
            data=[[Paragraph(inline_html(str(c)),styles['head']) for c in t['headers']]]
            data += [[Paragraph(inline_html(str(c)),styles['cell']) for c in row] for row in t['rows']]
            table=Table(data,colWidths=(widths*WIDTH).tolist(),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf1f3')),
                ('LINEABOVE',(0,0),(-1,0),.6,colors.HexColor('#314a58')),('LINEBELOW',(0,0),(-1,0),.4,colors.HexColor('#314a58')),
                ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#71858f')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding)]))
            story.append(KeepTogether([Paragraph(inline_html(t['caption']),styles['caption']),table,Spacer(1,8)]))
            md.extend(['**'+t['caption']+'**',markdown_table(t['headers'],t['rows'])])
            spec=''.join('>{\\raggedright\\arraybackslash}p{'+f'{w*.88:.4f}'+r'\textwidth}' for w in widths)
            tex.extend([r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{'+spec+'}',r'\caption*{'+inline_tex(t['caption'])+r'}\\',r'\toprule',' & '.join(inline_tex(str(c)) for c in t['headers'])+r' \\',r'\midrule',r'\endfirsthead',r'\toprule',' & '.join(inline_tex(str(c)) for c in t['headers'])+r' \\',r'\midrule',r'\endhead'])
            tex.extend(' & '.join(inline_tex(str(c)) for c in row)+r' \\' for row in t['rows'])
            tex.extend([r'\bottomrule',r'\end{longtable}',r'\endgroup'])
        elif token and token[1] in figures:
            file,caption=figures[token[1]]
            from PIL import Image as PILImage
            with PILImage.open(OUT/'figures'/file) as im:w,h=im.size
            drawing=Image(str(OUT/'figures'/file),width=WIDTH,height=WIDTH*h/w)
            story.append(KeepTogether([Spacer(1,6),drawing,Paragraph(inline_html(caption),styles['caption'])]))
            md.extend([f'![{caption}](figures/{file})'])
            tex.extend([r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width=\linewidth]{figures/'+file+'}',r'\caption*{'+inline_tex(caption)+'}',r'\end{figure}'])
        elif chunk.startswith('### '):
            heading=chunk[4:];story.append(Paragraph(inline_html(heading),styles['h2']));md.append(chunk);tex.append(r'\subsection{'+inline_tex(heading)+'}')
        elif chunk.startswith('## '):
            heading=chunk[3:];in_refs=heading=='References'
            if heading.startswith(('Appendix A','Appendix C','Appendix D')):
                story.append(PageBreak());tex.append(r'\clearpage')
            story.append(Paragraph(inline_html(heading),styles['h1']));md.append(chunk);tex.append(r'\section{'+inline_tex(heading)+'}')
        else:
            assert not token,'Unresolved token: '+str(token)
            story.append(Paragraph(inline_html(chunk.replace('\n',' ')),styles['small' if in_refs else 'body']))
            md.append(chunk);tex.append(inline_tex(chunk.replace('\n',' '))+'\n')
    tex.extend([r'\end{document}'])
    # caption* is provided by caption; add it once to the generated preamble.
    tex.insert(5,r'\usepackage{caption}')
    (OUT/'paper.md').write_text('\n\n'.join(md)+'\n',encoding='utf8')
    (OUT/'paper.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#a5b2ba'));canvas.setLineWidth(.4);canvas.line(54,38,A4[0]-54,38)
        canvas.setFont('Times-Roman',8);canvas.setFillColor(colors.HexColor('#40515e'))
        canvas.drawString(54,25,'Guo & Yang | Exploratory audit of coding-agent traces')
        canvas.drawRightString(A4[0]-54,25,str(doc.page))
    PDF.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=51,
        title=title,author='Yunjie Guo; Hanqing Yang',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    metrics=datasets();OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)
    ts=tables(metrics)
    for key,t in ts.items():pd.DataFrame(t['rows'],columns=t['headers']).to_csv(OUT/'tables'/f'{key.lower()}.csv',index=False)
    for name in ['fig_checkpoints.png','fig_within_issue.png']:shutil.copy2(RUN/'publication'/name,OUT/'figures'/name)
    figs={
        'FIG_CHECKPOINTS':('fig_checkpoints.png','Figure 1. Prespecified exploratory checkpoint contrasts and known-label non-audit coverage. Solid lines denote the main checkpoint-reaching population and dashed lines the fixed reached-30 cohort; bands show unadjusted paired repository-bootstrap percentile intervals. Population and information content change with K.'),
        'FIG_WITHIN':('fig_within_issue.png','Figure 2. Main command-presence odds ratios for failure, with successful-refit percentile intervals. Every displayed interval spans one. A: ordinary logistic; B: issue conditional; C: issue/configuration conditional. Logarithmic horizontal scale; vertical line denotes OR = 1. K15 Model C retains only 325/500 refits.')}
    template=(ROOT/'analysis_v2/final_paper_template.md').read_text(encoding='utf8')
    assert 'Yunjie Guo and Hanqing Yang' in template and 'Anonymous' not in template
    # Critical narrative values must agree with saved results to printed precision.
    for kind,pop,k,model,fields in [('logistic','main',15,'M2',['delta_ap','delta_ap_low','delta_ap_high']),('logistic','main',20,'M2',['delta_ap']),('logistic','fixed30',15,'M2',['delta_ap','delta_ap_low','delta_ap_high']),('hist_gradient_boosting','main',15,'M2',['delta_ap','delta_ap_low','delta_ap_high'])]:
        r=metrics[(metrics.kind==kind)&(metrics.population==pop)&(metrics.k==k)&(metrics.model==model)].iloc[0]
        for field in fields:assert num(r[field],9) in template,(kind,k,field,num(r[field],9))
    render(template,ts,figs)
    for name in ['checkpoint_metrics.csv','coverage.csv']:
        shutil.copy2(RUN/'checkpoints'/name,OUT/'tables'/name)
    for name in ['within_issue_odds_ratios.csv','within_issue_support.csv']:
        shutil.copy2(RUN/'within_issue'/name,OUT/'tables'/name)
    manifest=dict(authors=['Yunjie Guo','Hanqing Yang'],date='2026-11-03',language='English',new_model_scores_generated=False,
        template_sha256=sha(ROOT/'analysis_v2/final_paper_template.md'),builder_sha256=sha(Path(__file__)),
        source_workflow_sha256=sha(RUN/'WORKFLOW_STATUS.json'),source_measurement_lock_sha256=sha(MEAS/'AI_FINAL_LABEL_LOCK.json'),
        files={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in {'MANUSCRIPT_MANIFEST.json','README.md'}},
        pdf_path=str(PDF.relative_to(ROOT)),pdf_sha256=sha(PDF),all_new_results_exploratory=True)
    (OUT/'MANUSCRIPT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(dict(pdf=str(PDF),source=str(OUT),tables=len(ts),words=len((OUT/'paper.md').read_text().split())),indent=2))


if __name__=='__main__':main()
