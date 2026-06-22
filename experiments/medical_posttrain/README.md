# Medical Post-Training Experiment Plan

## 实验目标

围绕中文医疗大模型后训练，完成一条可解释、可复现、可面试追问的实验流水线。

重点不是追求最大模型或最长训练，而是证明：

- 理解 DPO/RM/PPO/GRPO 的训练目标和工程差异
- 能发现偏好数据质量问题
- 能设计数据清洗、prompt 重构和奖励函数
- 能通过日志、评测和案例形成闭环

## 实验 0：环境与数据

产物：

- 模型、数据下载命令
- 环境版本
- 数据字段统计
- 随机抽样样例

## 实验 1：DPO Baseline

输入：

- 原始 `medical/reward` 偏好数据
- 少量通用偏好数据，可选

记录：

- train loss
- reward margin
- chosen/rejected logprob 差异
- 5-10 个医疗问答案例

## 实验 2：偏好数据审计

检查：

- chosen/rejected 是否疑似反转
- chosen 是否包含格式错乱
- rejected 是否反而更专业
- prompt 是否过短或不完整
- 是否存在重复样本

脚本：

```bash
python experiments/medical_posttrain/data_audit.py \
  --input data/reward/medical_reward.jsonl \
  --output reports/data_audit_medical_reward.md
```

## 实验 3：DPO v2 数据重构

思路：

- 将口语化问题重构为临床病历式 prompt
- 统一字段：`system/history/prompt/chosen/rejected`
- 清洗明显反转和低质量样本
- 混入少量通用偏好样本，避免医疗过拟合

对比：

- DPO baseline vs DPO v2
- 训练曲线
- 医疗案例质量
- C-Eval 医学子集或自建 case 评估

## 实验 4：Reward Model

目标：

- 训练医疗偏好 reward model
- 评估 pairwise accuracy
- 为 PPO/GRPO 提供奖励信号或对比基线

## 实验 5：PPO Smoke Test

目标：

- 跑通 actor/reference/reward/critic
- 记录显存、速度、loss 变化
- 不追求大规模收敛

面试重点：

- PPO 为什么比 DPO 更重
- reference model 的 KL 约束作用
- reward hacking 如何观察

## 实验 6：GRPO Reward Design

奖励函数：

- 格式奖励：是否满足 `<think></think><answer></answer>`
- 语义相似度：答案与参考答案 embedding 相似度
- LLM judge：小模型或外部模型评分
- PPL 惩罚：惩罚胡言乱语和语言分布坍塌

目标：

- 小规模跑通
- 观察 reward hacking
- 调整 reward 权重

