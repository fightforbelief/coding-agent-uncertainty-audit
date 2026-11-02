"""Wait for verified features, evaluate frozen models, and generate paper tables.

Writes FAILED on any error rather than presenting partial runs as complete.
Stages resume from hash-checked outputs; no automatic test-set tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis_v2')]
import pandas as pd
from v3_5_ai_audit_report import md_table


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tex_table(frame):
    def esc(value):
        if pd.isna(value):return '--'
        if isinstance(value,float):return f'{value:.4g}'
        return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    rows=[r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\toprule',' & '.join(esc(c) for c in frame.columns)+r' \\',r'\midrule']
    rows.extend(' & '.join(esc(v) for v in row)+r' \\' for row in frame.itertuples(index=False,name=None))
    rows.extend([r'\bottomrule',r'\end{tabular}'])
    return '\n'.join(rows)


def report(output,measurement):
    model_dir=output/'checkpoints';within_dir=output/'within_issue';paper=output/'publication';paper.mkdir(exist_ok=True)
    primary=json.loads((model_dir/'PRIMARY_EXPLORATORY_RESULT.json').read_text(encoding='utf8'))
    p=primary['primary'];metrics=pd.read_csv(model_dir/'checkpoint_metrics.csv')
    measurement_frame=pd.read_csv(Path(measurement)/'final_ai_measurement.csv')
    pooled=measurement_frame[measurement_frame.scaffold.eq('all')]
    odds=pd.read_csv(within_dir/'within_issue_odds_ratios.csv')
    support=pd.read_csv(within_dir/'within_issue_support.csv')
    reliability=pd.read_csv(ROOT/'analysis_v2/original_plan/accepted_first_round_measurement_20261103_v2/first_round_reliability.csv')
    for source,name in [(model_dir/'fig_checkpoints.png','fig_checkpoints.png')]:shutil.copy2(source,paper/name)
    tables={
        'measurement':pooled[['frame','sample_memberships','positive_share_all','uncertain_share','positive_share_determinate']],
        'checkpoints':metrics[['kind','population','k','model','ap','auroc','brier','delta_ap','delta_ap_low','delta_ap_high']],
        'within_issue':odds,
        'support':support,
        'reliability':reliability[['window','field','sample_agreement','cohen_kappa']],
    }
    for name,table in tables.items():
        table.to_csv(paper/f'table_{name}.csv',index=False,encoding='utf-8-sig')
        (paper/f'table_{name}.tex').write_text(tex_table(table)+'\n',encoding='utf8')
    (paper/'tables.tex').write_text('\n\n'.join('% '+name+'\n'+tex_table(table) for name,table in tables.items()),encoding='utf8')
    interpretation=('新增代理特征的主要探索性 ΔAP 区间包含 0，增量未确立。' if p['delta_ap_low']<=0<=p['delta_ap_high'] else
                    ('主要探索性 ΔAP 为正且区间不含 0；这是当前冻结代理特征的预测比较，不能据此声称独立测量有效性、因果效应或实际用途已确立。' if p['delta_ap']>0 else '新增代理特征的主要探索性 ΔAP 为负，未显示预测增益。'))
    summary=f"""# P8 V3 结果汇总

本页数字均由结果文件生成。新分析因负责人选择 AI 辅助复核和 AI 裁决路线而保留为探索性；旧确认性结果保持原样。

主检查点 k*={int(p['k'])}，仅按 train+validation 覆盖率选择。测试集纳入 {int(p['n_test']):,} 条 known-label 非审计轨迹。主要探索性 M2−M0 ΔAP={p['delta_ap']:.9g}，repository-cluster paired bootstrap 95% 区间 [{p['delta_ap_low']:.9g}, {p['delta_ap_high']:.9g}]。{interpretation}

来源：`../checkpoints/PRIMARY_EXPLORATORY_RESULT.json`、`../checkpoints/checkpoint_metrics.csv`。

审计样本 1,196 条已从所有划分排除。AI 裁决采用 2,321 个候选字段，并记录 225 个窗口中 506 项额外逻辑一致性修改；不冒充第三位人类独立裁决。测量点估计来源：`table_measurement.csv` 与原 `AI_FINAL_LABEL_LOCK.json`。首轮一致性来源：`table_reliability.csv`；一致性不是准确率。

主分析和固定 reached_30 人群均给出全部五个检查点；不能用某个次要 k 的区间代替主要探索性终点。长轨迹条件会改变人群，固定人群敏感性用于显示这种差别，不能消除行为测量误差或反向时序。

Issue 内比较在 k=10/15 使用全部 known-label 非审计轨迹；A/B/C 及分配置结果见 `table_within_issue.csv`，支持度见 `table_support.csv`。OR 的阳性结局是任务失败 resolved==0。所有关联均为观察性，不能解释成跑测试改变成功率。

非线性基线只用验证集选参数，单独报告为次要分析。bootstrap 失败数、不可识别变量、支持不足和收敛状态保留在每个模型 JSON 中，不将失败替换成 0。

旧冻结 ΔAP 为 0.00006589172683257871，区间 [0.00000560783009556365, 0.00013282286584345433]；新结果不能用于改写旧确认性结论。完整偏离见项目 `analysis_v2/DEVIATIONS.md`，冻结协议见 `analysis_v2/PROTOCOL_v3.md`。
"""
    (paper/'SUMMARY.md').write_text(summary,encoding='utf8')
    text=f"""# Early Test-Command Signals in Coding-Agent Traces: An Exploratory Measurement and Predictive Audit

## Abstract

We audited the pinned Open-SWE-Traces snapshot (revision c144d9d160dcd9f5ebcd9520d2d54111dda8b681; 207,489 trajectories) using fixed prefix rules, a 1,196-trajectory stratified audit, repository-held-out prediction, and within-issue comparisons. The submitted review drafts were AI-assisted and reported by the project owner as individually human-reviewed; the owner subsequently selected disclosed AI adjudication. This is not an independently human-adjudicated gold standard. Audit trajectories were excluded from every model split. The preselected primary exploratory checkpoint was {int(p['k'])} completed rounds, selected using train/validation coverage only. On {int(p['n_test']):,} scored trajectories, adding the prespecified behavior proxies produced delta average precision {p['delta_ap']:.9g} with a paired repository-cluster bootstrap 95% interval [{p['delta_ap_low']:.9g}, {p['delta_ap_high']:.9g}]. All new findings remain exploratory and do not establish causality or practical usefulness.

## Design and provenance

The historical repository-connected-component split and original structural feature definitions were retained. The outcome is failure (resolved==0), with unknown outcomes excluded. Fixed checkpoints were 5, 10, 15, 20 and 30 completed rounds. M0 contains the original structural features at the corresponding prefix; M1 adds a strict failure proxy; M2 adds detected test-command presence, log1p count, failure and same-runner failure-then-pass. Detector rules were not tuned on the new test results. The primary exploratory contrast was fixed before scoring as M2−M0 at k*. Logistic regularization was chosen by validation AP, then models were refitted on train+validation. Test prediction vectors were saved once per prespecified scoring cell. Confidence intervals resampled repositories jointly for both compared predictions (1,000 draws; seed 20261092).

## Measurement audit

The audit follows the frozen V3.5 stratum memberships and their own N_h/n_h weights. Main and diagnostic strata are not pooled; overlapping inclusion probabilities are never summed. Both submitted first rounds, their disagreements, provenance and the owner authorization are retained. The final AI-reconciled measurement table is descriptive of that declared source. Its determinate-label ratios must not be represented as gold-standard human accuracy or recall. Ambiguity bounds are classification sensitivity ranges, not confidence intervals. Original rule/command, process, test execution and test result categories remain distinct.

{md_table(tables['measurement'])}

## Predictive results

{md_table(metrics[(metrics.kind=='logistic')&metrics.population.eq('main')][['k','model','ap','auroc','brier','delta_ap','delta_ap_low','delta_ap_high']])}

![Checkpoint contrasts and coverage](fig_checkpoints.png)

The fixed reached_30 comparison and nonlinear baselines are supplied in the machine-generated tables. Conditioning on reaching each prefix changes the population; the fixed cohort controls this aspect of comparison but cannot remove measurement error or time-dependent selection. Additional checkpoints, nonlinear models and metrics are secondary exploratory analyses, not opportunities to replace the primary contrast.

## Within-issue associations

Ordinary, issue-conditional and—when supported—issue/configuration-conditional logistic comparisons were prespecified at 10 and 15 rounds. Outcome-constant groups do not identify conditional effects. Covariates without within-group variation are recorded as unidentified rather than forced into a model. Repository-cluster bootstrap refits (500 requested) provide intervals, with failed refits reported. The exact conditional likelihood and score are checked against statsmodels; multiplicity weights repeat complete conditional likelihood terms when a repository is drawn repeatedly. The odds-ratio tables retain subgroup analyses, support counts and convergence limitations. Positive coefficients indicate association with failure, not success.

## Limitations and interpretation

This hypothesis was partly motivated by earlier test-set observations, and the same historical snapshot is reused. The project-owner-approved AI adjudication deviates from the original third-human-adjudicator route. Draft origins, approximate review dates and the owner's independence/time-line attestation are disclosed; the software does not authenticate identities or manufacture review timestamps. High agreement or favorable predictive metrics cannot establish that the behavioral proxies perfectly measure actual test execution. Errors shared by automated rules, unknown commands, short prefixes and output-format ambiguities remain possible. Agents may test after locating a problem, creating reverse temporal interpretation. No causal effect, external generalization, novelty or deployment benefit is established by these analyses.

The original frozen study's very small positive incremental AP is preserved as a separate historical result. The present work reports its prespecified exploratory comparisons without retroactively relabelling them as an independently confirmed finding.

## Reproducibility

All numerical tables are generated from saved results. Per-cell predictions, validation choices, bootstrap outputs, repository exclusions, feature hashes, the frozen protocol and deviation log are retained. Skipped/failed conditional fits are explicit. No missing or failed result is imputed as a zero effect.
"""
    (paper/'paper_v3.md').write_text(text,encoding='utf8')
    return dict(primary=primary['primary'],publication=str(paper),files={p.name:sha(p) for p in paper.iterdir() if p.is_file()})


def run(feature_dir,measurement,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    status=output/'WORKFLOW_STATUS.json'
    def save(stage,**kwargs):status.write_text(json.dumps(dict(stage=stage,**kwargs),indent=2),encoding='utf8')
    try:
        deadline=time.time()+4*3600;last=0
        while not (Path(feature_dir)/'FEATURE_RUN_MANIFEST.json').exists():
            if time.time()>deadline:raise TimeoutError('Feature extraction did not complete within the declared wait window')
            if time.time()-last>=60:
                completed=len(list(Path(feature_dir).glob('data__*.json')))
                save('WAITING_FOR_FEATURES',completed_shards=completed,total_shards=84)
                print(json.dumps(dict(stage='waiting_features',completed_shards=completed,total_shards=84)),flush=True);last=time.time()
            time.sleep(10)
        feature_manifest=json.loads((Path(feature_dir)/'FEATURE_RUN_MANIFEST.json').read_text(encoding='utf8'))
        if feature_manifest['status']!='COMPLETE':raise ValueError('Features did not pass final verification')
        features=Path(feature_dir)/'features_v3.parquet'
        if sha(features)!=feature_manifest['features_sha256']:raise ValueError('Feature hash changed')
        save('CHECKPOINT_MODELS')
        subprocess.run([sys.executable,str(ROOT/'analysis_v2/v3_checkpoint_models.py'),'--features',str(features),'--bindings',str(ROOT/'analysis_v2/V3_PROTOCOL_BINDINGS.json'),
                        '--exclusions',str(ROOT/'analysis_v2/audit/audit_ids_EXCLUDE.txt'),'--output',str(output/'checkpoints')],check=True)
        save('WITHIN_ISSUE_MODELS')
        subprocess.run([sys.executable,str(ROOT/'analysis_v2/v3_within_issue.py'),'--features',str(features),'--exclusions',str(ROOT/'analysis_v2/audit/audit_ids_EXCLUDE.txt'),
                        '--output',str(output/'within_issue')],check=True)
        save('GENERATING_PUBLICATION')
        result=report(output,measurement)
        save('COMPLETE',all_new_results_exploratory=True,**result)
        print(json.dumps(dict(stage='COMPLETE',**result)),flush=True)
    except BaseException as error:
        save('FAILED',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--feature-dir',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.feature_dir,a.measurement,a.output)
