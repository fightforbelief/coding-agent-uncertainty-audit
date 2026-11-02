# V3 后续分析协议（2026-11-03，测试集评分前冻结）

本协议记录项目负责人批准继续后的计算方案。负责人选择 AI 候选裁决代替原第三位人类裁决，首轮判断亦来自 AI 辅助草稿后据负责人陈述的人类复核。来源与方案偏离见 DEVIATIONS.md。因此本轮是**事先指定计算口径的探索性扩展**，不声称独立人工测量验证完成，不将新结果升级为未受历史测试集观察影响的确认性证据。

研究问题部分受原测试集 7 条阳性轨迹观察启发。旧论文与旧确认性结果保持原样。新研究使用冻结 detector 的代理变量，不声称真实测试执行已被完美测量，也不作因果解释。

## 数据、隔离与主检查点

- Revision：`c144d9d160dcd9f5ebcd9520d2d54111dda8b681`；沿用 repository-connected-component 划分 seed 20261024。
- 只对 `resolved` 为 0/1 的轨迹建模，预测阳性为 `resolved==0`。
- 从所有 train/validation/test 和解释性分析中剔除 `audit/audit_ids_EXCLUDE.txt` 的全部 1,196 条审计轨迹；该名单已在新评分前写定，SHA-256 见 `audit/EXCLUSION_SET_MANIFEST.json`。
- 检查点为 5、10、15、20、30。主检查点从 15、10、5 中选择 train+validation known-label 非审计轨迹上 reached_k 比例至少 70% 的最大值；不使用测试集决定。选择结果与 detector/源代码哈希另存 `V3_PROTOCOL_BINDINGS.json`。
- 主分析每个 k 只纳入 reached_k==1；固定人群敏感性只纳入 reached_30==1，在相同人群上比较全部 k。

## 特征与模型

使用冻结 parser 和 `p8v3/testcmd.py`，不改检测规则。结构特征复用原 `_features_at_checkpoint` 的 `STRUCTURAL_FIELDS`，仅改变 K。提取只需冻结完成轮次边界以内的前 30 轮；短轨迹保留完整可用轨迹。检测器的调用窗口仍是 round<=K，未完成尾部调用不会被计作额外完成轮次。累计类特征必须单调；结构特征必须与全轨迹调用原函数一致。

- M0：该 K 的原结构特征。
- M1：M0 + high-confidence 测试调用的显式失败标记。
- M2：M0 + ran-tests + log1p(test-run count) + test-fail + fail-then-pass。
- fail-then-pass：窗口内先有 high-confidence 调用的显式失败，随后同一 runner 有明确通过汇总且该次没有失败。
- `no_test` 是 ran-tests 的补值；首次测试轮次和缺失命令数另存诊断，不作为 M0/M1/M2 额外选择的特征。
- L2 logistic：C∈{0.01,0.1,1,10}，lbfgs，max_iter=5000。验证集 AP 选 C，平局选较小 C；训练阶段标准化器只在 train 拟合，最终与模型一起在 train+validation 重拟合。使用与旧 L2 同一模型帮助函数。

## 评价

事先指定的主要探索性终点是主 k 的 AP(M2)-AP(M0)。次要：M1-M0、其余 k、AUROC、Brier、随机 AP 基线、所选 C、M2 系数。区间使用同一组 repository-cluster 重抽样作成对差值，seed 20261092，1,000 次，95% 百分位区间；报告有效/跳过重复数。区间含 0 写“未确立”。不事后更换主 k、C 网格、标签或首要终点。

各个预指定评分单元只评一次并保存逐行预测；重跑验收只读保存预测，避免用测试集反复调参。实现错误需停止并记录偏离，而不是依据测试分数修改算法。按 K/模型的多个次要分析不作确认性显著性解释。

## 解释性 issue 内比较

K=10、15，全部 known-label 非审计轨迹。先描述 issue × scaffold/model 的结局与行为支持度。

- A：普通 logistic，测试行为四项，加 log1p(tool-call count)、distinct-tool count、log1p(observation chars)、scaffold/model 配置哑变量。
- B：ConditionalLogit，groups=issue，保留配置哑变量。只有结局不同的组贡献似然。
- C：若同 (issue,scaffold,model) 中结局不同的组不少于 500，则以该组作 ConditionalLogit，去掉配置哑变量；否则明确报告支持不足，不强行拟合。
- 组内完全不变的协变量不进入条件模型，逐项记录；不通过结果大小选变量。报告收敛状态和有效组数。
- OR 区间使用 repository-cluster bootstrap，500 次，seed 20261092，每次重拟合；重复抽到仓库时为组 ID 加重复编号，不能把重复抽样副本合并成同一个条件组。报告不收敛次数。按配置分层作为次要分析。
- 所有 OR 均为观察关联，不作因果效应。测试行为可能发生在已定位问题之后，存在反向时序。

## 补充基线与汇总

原主分析的 AUROC、Brier、10 等宽分箱校准已单独完成。新增非线性次要基线：HistGradientBoostingClassifier，在主 k 与 k=5 上对 M0/M2 使用同样的非审计划分。max_leaf_nodes∈{15,31}、learning_rate∈{0.05,0.1}，max_iter=100、early_stopping=True、random_state=20261103。仅验证集 AP 选参数，平局取枚举顺序较前者，再在 train+validation 重拟合并一次性保存测试预测；成对 bootstrap 与上述相同。

表格与图均由结果文件生成，不手抄新结果。最终报告区分旧冻结主结果、本次事先指定的主要探索性终点和其他解释性/敏感性分析。没有独立人类裁决的测量限制始终保留。

本协议及后续冻结提交在任何新 V3 测试评分前写定。提交哈希写入独立绑定文件，不通过自引用伪造一个文件自身的 Git 哈希。
