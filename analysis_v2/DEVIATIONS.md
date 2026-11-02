# V3 方案偏离记录

1. 早期 200 条 P/F/S/R 方案已在既有 V3.5/V3.5.1 设计中改为固定的 1,196 条 A/B/C/D/AMB 审计包。后续排除集合使用全部 1,196 条，不混用旧样本。
2. 提交文件原为 AI 辅助草稿；负责人声明 Yunjie Guo 与 Hanqing Yang 已各自逐条复核，之后认可时间线解释并声明没有互看。原始文件和近似日期仍保留，不填造精确 UTC 人工时间，也不声称工具验证了身份或时间线。
3. 负责人明确选择“采用已生成的 AI 候选裁决，并明确记录 AI 来源和方案偏离”。本轮采用 2,321 个候选字段，来源为先前冻结的自动证据标注，不冒充第三位人类裁决。
4. 逐字段拼接后，225 个窗口出现最终无测试命令但执行/输出字段仍适用的逻辑组合。按规则将相应 506 个字段置为 not_applicable，每个额外变更均单独记为 AI 逻辑一致性修正，原两份首轮不改。
5. 最终测量是负责人接受的 AI 辅助复核及 AI 裁决结果，不是独立人类裁决的金标准。首轮分歧和不确定性另报；不据行政“全部通过”宣称检测器 accuracy、recall 或测量质量被统计证明。
6. 人工测量计划原未选择正式 issue 聚类区间方法。没有事后补造一个确认性测量区间；分类敏感性范围不称为置信区间。
7. 新研究问题部分来自旧测试集观察，且测量裁决偏离原人工路线。后续模型计算虽在新评分前冻结，所有新增推断仍标为探索性，不重新包装为独立确认性验证。旧论文和旧确认性数值不改。
8. 为完成预定 500 次仓库 bootstrap，条件 logistic 使用矢量化精确条件似然/梯度优化，并在合成样本及每个实际点估计上与 statsmodels ConditionalLogit 核对。按仓库的整数重复权重重复整组似然，与给重抽样副本分别编号等价，不把重复行塞入同一个扩大的条件组。该实现优化不改变估计对象；不以渐近标准误取代预定的 bootstrap 区间。
9. 全库特征提取使用相同冻结 worker 的正向/反向缓存并行与逐字节缓存核对。它只改变计算顺序；84 个原始分片仍各自核验哈希。源 detector、parser、结构特征和检查点定义不改变。

后续实现错误、数据不一致或收敛问题必须在发生时追加，不能根据测试分数选择性删除。

10. Before any new scoring, independent within-issue analysis cells were scheduled across four local worker processes. Each cell retains its fixed seed and all 500 repository-bootstrap refits; no likelihood, estimator, covariate or interval rule changed. Numerical libraries use one thread per worker. This changes computation scheduling only.
