"""Render verified completed V3 results as a compact scientific PDF report."""
import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether

ROOT=Path(__file__).resolve().parents[1]
INK=colors.HexColor('#18313d');ACCENT=colors.HexColor('#287c8e');PALE=colors.HexColor('#edf4f6')
WIDTH=A4[0]-100


def number(value,digits=5):
    if value is None or pd.isna(value):return '--'
    return f'{value:.{digits}g}'


def forest(odds,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sub=odds[odds.analysis.isin(['A','B','C'])&odds.feature.eq('ran_tests')].sort_values(['k','analysis'])
    fig,ax=plt.subplots(figsize=(7,3.1),layout='constrained')
    labels=[]
    for position,row in enumerate(sub.itertuples(index=False)):
        labels.append(f'K={row.k}, model {row.analysis}')
        if pd.isna(row.odds_ratio) or row.odds_ratio<=0:continue
        color={'A':'#b6634e','B':'#287c8e','C':'#49766a'}[row.analysis]
        ax.plot(row.odds_ratio,position,'o',color=color)
        if not pd.isna(row.or_low) and not pd.isna(row.or_high) and row.or_low>0:
            ax.plot([row.or_low,row.or_high],[position,position],color=color)
    if not len(sub):ax.text(.5,.5,'Not estimable; see per-analysis status files',ha='center',transform=ax.transAxes)
    ax.set_yticks(np.arange(len(labels)),labels);ax.invert_yaxis();ax.axvline(1,color='grey',linestyle='--',linewidth=.8)
    ax.set_xscale('log');ax.set_xlabel('Failure odds ratio for detected test-command presence')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(target,dpi=180);plt.close(fig)


def create(analysis,measurement,output,synthetic=False):
    analysis,measurement,output=Path(analysis),Path(measurement),Path(output)
    state=json.loads((analysis/'WORKFLOW_STATUS.json').read_text(encoding='utf8'))
    if state.get('stage')!='COMPLETE':raise ValueError('Never publish a partial/failed workflow as a final PDF')
    if state.get('is_synthetic_fixture') and not synthetic:raise ValueError('Synthetic fixtures cannot be published as real study results')
    if synthetic and 'tmp' not in str(output).lower():raise ValueError('Synthetic layout tests must stay in tmp')
    model=analysis/'checkpoints';within=analysis/'within_issue';pub=analysis/'publication'
    p=json.loads((model/'PRIMARY_EXPLORATORY_RESULT.json').read_text(encoding='utf8'))['primary']
    metrics=pd.read_csv(model/'checkpoint_metrics.csv')
    measure=pd.read_csv(measurement/'final_ai_measurement.csv');measure=measure[measure.scaffold.eq('all')]
    odds=pd.read_csv(within/'within_issue_odds_ratios.csv')
    fits=[json.loads(p.read_text(encoding='utf8')) for p in within.glob('*_k*.json')]
    completed=[r for r in fits if r['status']=='COMPLETE']
    failed_refits=sum(len(r.get('bootstrap_failed',[])) for r in completed)
    requested_refits=sum(r['bootstrap_requested'] for r in completed)
    minimum_valid=min((r['bootstrap_valid'] for r in completed),default=0)
    source_reliability=ROOT/'analysis_v2/original_plan/accepted_first_round_measurement_20261103_v2/first_round_reliability.csv'
    rel=pd.read_csv(source_reliability);rel=rel[rel.field.eq('test_command_status')]
    pub.mkdir(exist_ok=True);forest(odds,pub/'fig_within_issue.png')
    output.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyAudit',fontName='Helvetica',fontSize=9.4,leading=13.7,textColor=INK,spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallAudit',fontName='Helvetica',fontSize=8,leading=11,textColor=INK,spaceAfter=7))
    styles.add(ParagraphStyle(name='TitleAudit',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=INK,spaceAfter=15))
    styles.add(ParagraphStyle(name='SectionAudit',fontName='Helvetica-Bold',fontSize=15,leading=20,textColor=ACCENT,spaceAfter=12))
    styles.add(ParagraphStyle(name='CellAudit',fontName='Helvetica',fontSize=8.2,leading=10.8,textColor=INK))
    story=[]
    def para(text,small=False):return Paragraph(text,styles['SmallAudit' if small else 'BodyAudit'])
    def add(text,small=False):story.append(para(text,small))
    def heading(text):story.append(Paragraph(text,styles['SectionAudit']))
    def table(headers,rows,widths=None):
        data=[[Paragraph(escape(str(v)),styles['CellAudit']) for v in headers]]
        data.extend([Paragraph(escape(str(v)),styles['CellAudit']) for v in row] for row in rows)
        t=Table(data,colWidths=widths or [WIDTH/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('LINEBELOW',(0,0),(-1,0),.7,ACCENT),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#bacbd2'))]))
        story.extend([t,Spacer(1,12)])
    def interval(low,high):return '['+number(low,4)+', '+number(high,4)+']'
    def picture(path,width=WIDTH):
        from PIL import Image as PILImage
        with PILImage.open(path) as image:w,h=image.size
        story.extend([Image(str(path),width=width,height=width*h/w,hAlign='LEFT'),Spacer(1,8)])
    def page():story.append(PageBreak())

    if synthetic:add('<b>SYNTHETIC LAYOUT TEST - NOT RESEARCH RESULTS</b>')
    add('P8 / RESEARCH AUDIT / 03 NOVEMBER 2026',True)
    story.append(Paragraph('Early Test-Command Signals<br/>in Coding-Agent Traces',styles['TitleAudit']))
    add('Repository-held-out prediction, measurement audit and within-issue analysis',False)
    heading('1. Main result and scope')
    add('This report extends the pinned Open-SWE-Traces study with fixed early-prefix behavior proxies. The data snapshot contains 207,489 trajectories. A fixed 1,196-trajectory audit is excluded from every train, validation and test analysis. The preselected primary exploratory checkpoint is '+str(int(p['k']))+' completed rounds, chosen using train/validation coverage only.')
    key='The interval includes zero; an incremental AP was not established.' if p['delta_ap_low']<=0<=p['delta_ap_high'] else ('A positive increment was observed for these proxy features.' if p['delta_ap']>0 else 'The added proxy features reduced AP in this comparison.')
    add('<b>Primary exploratory comparison:</b> M2 minus M0 Delta AP = <b>'+number(p['delta_ap'],7)+'</b>; paired repository-cluster bootstrap 95% interval '+interval(p['delta_ap_low'],p['delta_ap_high'])+'. '+key)
    selected=metrics[(metrics.kind=='logistic')&metrics.population.eq('main')&metrics.k.eq(p['k'])]
    table(['Model','Average precision','AUROC','Brier'],[[r.model,number(r.ap,6),number(r.auroc,6),number(r.brier,6)] for r in selected.itertuples(index=False)],[65,145,140,WIDTH-350])
    add('Scored trajectories: '+format(int(p['n_test']),',')+'. Positive outcome means failure (resolved==0). M0 uses the original prefix structure; M1 adds a strict failure proxy; M2 adds detected test-command presence, log-count, failure and same-runner failure-then-pass.')
    add('<b>Interpretation boundary.</b> All new findings are exploratory. Review inputs were AI-assisted and reported by the project owner as individually human-reviewed. The owner selected disclosed AI adjudication instead of a third human adjudicator. This is not an independently human-adjudicated gold standard, a causal study or evidence of deployment usefulness.')
    add('The original frozen result is retained separately: Delta AP 0.0000658917, historical 95% interval [0.00000560783, 0.000132823]. New comparisons use different prefixes and audit exclusions; they are not a direct before/after comparison to that historical score.',True)
    page()

    heading('2. Measurement and source provenance')
    add('The frozen audit has 1,196 unique trajectories and 1,197 stratum memberships. Each estimate uses its own N_h/n_h weight. Main positive, known-negative, missing-command and ambiguous-command frames remain separate. The one cross-frame trajectory does not acquire a summed inclusion probability.')
    short={'R5_detector_positive_frame':'R5 detector-positive','R15_detector_positive_frame':'R15 detector-positive',
        'R15_detector_known_negative_frame':'R15 known-negative','R15_missing_command_diagnostic':'R15 missing-command',
        'R5_ambiguous_command_diagnostic':'R5 ambiguous-command','R15_ambiguous_command_diagnostic':'R15 ambiguous-command'}
    table(['Frame','Sample n','Positive share','Uncertain share','Determinate positive'],
        [[short[r.frame],int(r.sample_memberships),number(r.positive_share_all*100,4)+'%',number(r.uncertain_share*100,4)+'%',number(r.positive_share_determinate*100,4)+'%' if pd.notna(r.positive_share_determinate) else '--'] for r in measure.itertuples(index=False)],
        [145,55,95,95,WIDTH-390])
    add('Percentages describe the AI-reconciled labels within each frame, not overall behavior prevalence or gold-standard accuracy. The known-negative frame denominator is detector-negative trajectories, so its positive share is not a false-negative rate or recall. Unknowns and reviewer disagreements are not silently counted as negatives.',True)
    add('The original submitted files are preserved. The owner accepted the timeline explanation, stated no mutual viewing and selected AI candidate adjudication. Exactly 2,321 candidate fields were adopted; a further 506 fields in 225 windows were changed to not_applicable because a final no-test-command decision cannot retain applicable test-execution/output fields. Every additional change is logged as AI logical consistency, not a human judgement.')
    table(['Window','First-round sample agreement','Cohen kappa'],[[r.window,number(r.sample_agreement*100,4)+'%',number(r.cohen_kappa,4)] for r in rel.itertuples(index=False)],[80,245,WIDTH-325])
    add('First-round agreement is measured on the enriched audit sample, not the full population. Software does not authenticate reviewer identities, reconstruct precise review timestamps or erase the original AI-draft provenance. No formal measurement confidence interval was invented after seeing the labels.',True)
    page()

    heading('3. Checkpoint prediction and coverage')
    add('The historical repository-connected-component split is retained. C in {0.01, 0.1, 1, 10} is chosen by validation AP, with ties taking the smaller value; the model and scaler are then refitted on train plus validation. Each prespecified scoring cell saves its prediction vectors once. Bootstrap uses paired repository draws, 1,000 requested, seed 20261092.')
    picture(model/'fig_checkpoints.png',WIDTH-10)
    sub=metrics[(metrics.kind=='logistic')&metrics.population.eq('main')&metrics.model.eq('M2')].sort_values('k')
    table(['K','Test n','M2 AP','Delta AP','95% paired interval'],[[int(r.k),int(r.n_test),number(r.ap,5),number(r.delta_ap,5),interval(r.delta_ap_low,r.delta_ap_high)] for r in sub.itertuples(index=False)],[30,65,90,90,WIDTH-275])
    add('Checkpoint-specific inclusion changes the population. Fixed reached_30 sensitivity keeps the same population across K, but does not remove measurement error or reverse temporal interpretation. Secondary K values cannot replace the primary contrast.',True)
    page()

    heading('4. Fixed-population and nonlinear sensitivity')
    add('Fixed-population results restrict every checkpoint to trajectories reaching 30 completed rounds. The comparisons remain paired within each scoring population; AP levels across different populations also reflect changes in class prevalence and task composition.')
    sub=metrics[(metrics.kind=='logistic')&metrics.population.eq('fixed30')&metrics.model.eq('M2')].sort_values('k')
    table(['K','Test n','M2 AP','Delta AP','95% paired interval'],[[int(r.k),int(r.n_test),number(r.ap,5),number(r.delta_ap,5),interval(r.delta_ap_low,r.delta_ap_high)] for r in sub.itertuples(index=False)],[30,65,90,90,WIDTH-275])
    add('<b>Nonlinear secondary baseline.</b> HistGradientBoostingClassifier uses max_leaf_nodes in {15,31}, learning_rate in {0.05,0.1}, max_iter=100, early_stopping=True and random_state=20261103. Only validation AP chooses parameters. These are secondary analyses, not new primary endpoints.')
    sub=metrics[metrics.kind.eq('hist_gradient_boosting')].sort_values(['k','model'])
    table(['K','Model','AP','Delta AP','95% paired interval'],[[int(r.k),r.model,number(r.ap,5),number(r.delta_ap,5),interval(r.delta_ap_low,r.delta_ap_high) if pd.notna(r.delta_ap) else '--'] for r in sub.itertuples(index=False)],[30,65,90,90,WIDTH-275])
    add('Complete M0/M1/M2 metrics, random AP baselines, parameter selections and standardized coefficients are supplied in the CSV/JSON tables. Null or failed results are not replaced by zero effects.')
    page()

    heading('5. Within-issue observational associations')
    add('Models A (ordinary logistic), B (issue conditional logistic) and C (issue/configuration conditional logistic when at least 500 outcome-discordant cells exist) are evaluated at K=10 and 15 on all known-label non-audit trajectories. Configuration indicators are included where identified. Predictors without within-group variation are explicitly dropped, rather than assigned an effect.')
    picture(pub/'fig_within_issue.png',WIDTH-5)
    sub=odds[odds.analysis.isin(['A','B','C'])&odds.feature.eq('ran_tests')].sort_values(['k','analysis'])
    table(['K','Model','Failure OR','95% interval','Valid bootstrap'],[[int(r.k),r.analysis,number(r.odds_ratio,5),interval(r.or_low,r.or_high),int(r.bootstrap_valid)] for r in sub.itertuples(index=False)],[30,55,95,180,WIDTH-360])
    add('A positive log coefficient or OR above 1 denotes association with failure, not success. The exact conditional likelihood and score are checked against statsmodels. Repository multiplicities repeat whole likelihood terms; they do not merge duplicated rows into enlarged conditional groups. Five hundred bootstrap refits are requested per analysis, and failed/rank-deficient refits are reported.',True)
    add('Agents may test after locating a problem. Neither these within-issue associations nor held-out predictive increments identify a causal effect of running tests. Configuration-specific results and all behavior coefficients are in the accompanying tables.',True)
    if not synthetic:
        add(f'<b>Refit stability:</b> {failed_refits:,} of {requested_refits:,} requested refits failed or lacked rank; the least stable analysis retained {minimum_valid}/500. Intervals use successful refits only. High failure rates can distort percentile intervals, so nominal 95% coverage is not established for those subgroups. Full per-analysis counts accompany the report.',True)
    page()

    heading('6. Limitations and reproducibility')
    for text in [
        '<b>Exploratory reuse.</b> The new hypothesis was partly motivated by earlier test-set observations on the same historical snapshot. A frozen scoring plan prevents later endpoint switching, but does not make this an independent confirmatory replication.',
        '<b>Measurement.</b> AI-assisted first-round provenance and owner-selected AI adjudication remain substantive limitations. High determinate-label agreement does not prove that proxies capture actual test execution. Unknown commands, custom scripts, output formats and errors shared by rules may remain.',
        '<b>Selection.</b> Reaching a checkpoint conditions on trajectory duration. The fixed-cohort analysis changes one aspect of the comparison, but cannot remove time-dependent task difficulty, model/configuration effects or reverse temporal ordering.',
        '<b>Intervals and convergence.</b> Prediction intervals resample repositories jointly for compared models; conditional-model intervals refit after repository resampling. Failed or unsupported estimates are explicit in the per-model files. Measurement ambiguity bounds are not confidence intervals.',
        '<b>Data protection.</b> Audit identities are excluded from all model partitions. All 207,489 K5 structural feature rows must exactly match the frozen historical table before any new prediction score. Each original shard is checked against its pinned SHA-256 before extraction.',
        '<b>Frozen plan.</b> The protocol was committed locally before new scoring (initial protocol commit e104202). The implementation and numerical checks were also frozen before scoring. Original source submissions, owner attestations, AI candidate provenance and logical consistency changes are retained.',
        '<b>Reproducible artifacts.</b> PROTOCOL_v3.md; DEVIATIONS.md; FEATURE_RUN_MANIFEST.json; MODEL_RUN_CONTEXT.json; MODEL_RUN_MANIFEST.json; WITHIN_ISSUE_MANIFEST.json; AI_FINAL_LABEL_LOCK.json; saved per-cell predictions and bootstrap outputs; machine-generated CSV and LaTeX tables.',
    ]:add(text)
    add('Dataset: nvidia/Open-SWE-Traces, pinned revision c144d9d160dcd9f5ebcd9520d2d54111dda8b681. The original paper and its historical small positive increment are preserved separately. No claim of novelty, causality or practical deployment benefit follows automatically from this report.',True)
    def decorate(canvas,doc):
        canvas.setStrokeColor(ACCENT);canvas.setLineWidth(.7);canvas.line(50,37,A4[0]-50,37)
        canvas.setFont('Helvetica',8);canvas.setFillColor(INK);canvas.drawString(50,24,'P8 | Exploratory audit | 03 Nov 2026');canvas.drawRightString(A4[0]-50,24,str(doc.page))
        if synthetic:
            canvas.saveState();canvas.setFillColor(colors.HexColor('#d6dddd'));canvas.setFont('Helvetica-Bold',35);canvas.translate(A4[0]/2,A4[1]/2);canvas.rotate(35);canvas.drawCentredString(0,0,'SYNTHETIC LAYOUT TEST');canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=49,
                         title='Early Test-Command Signals in Coding-Agent Traces',author='AI-assisted research audit')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic',action='store_true')
    a=p.parse_args();print(create(a.analysis,a.measurement,a.output,a.synthetic))
