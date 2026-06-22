# MedicalGPT Post-Training Reproduction

本仓库用于把 MedicalGPT 项目整理成可复现、可写进秋招简历的大模型后训练项目。

当前目标不是重跑已经丢失的 PT/SFT 产物，而是基于公开模型与公开数据，完成一条可验证的医疗后训练流水线：

1. DPO baseline
2. 偏好数据审计与清洗
3. DPO v2 数据重构实验
4. Reward Model 训练与 pairwise 评估
5. PPO smoke test
6. GRPO 奖励函数设计与小规模验证

## 推荐目录

```text
MedicalGPT/
├── configs/                    # 本项目自己的训练配置
├── docs/                       # AutoDL / VSCode / Codex 工作流说明
├── experiments/medical_posttrain/
│   ├── README.md               # 实验路线
│   ├── data_audit.py           # 偏好数据审计脚本
│   ├── eval_samples.jsonl      # 手工评测样例
│   └── reward_design.md        # GRPO/RL 奖励设计
├── reports/
│   └── experiment_log.md       # 实验记录模板
└── scripts/autodl/             # AutoDL 远程运行脚本
```

官方 MedicalGPT 源码建议放在当前仓库根目录，或放在相邻目录后通过环境变量 `MEDICALGPT_ROOT` 指定。

## 运行文档

- [AutoDL + VSCode + Codex 工作流](docs/autodl_workflow.md)
- [MedicalGPT 源码接入](docs/source_setup.md)
- [后训练执行手册](docs/runbook.md)
- [实验路线](experiments/medical_posttrain/README.md)
- [GRPO 奖励设计](experiments/medical_posttrain/reward_design.md)

## 不下载模型的源码克隆命令

如果 Codex 里网络不稳定，可以在本机终端执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none https://github.com/shibing624/MedicalGPT.git /Users/kristianzeng/Documents/MedicalGPT-src
```

如果 `--filter=blob:none` 失败，改用：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/shibing624/MedicalGPT.git /Users/kristianzeng/Documents/MedicalGPT-src
```

## 项目证据

最终需要保存：

- 训练命令与配置
- 数据审计报告
- DPO v1/v2 对比
- RM pairwise accuracy
- PPO/GRPO 运行日志
- 医疗问答推理样例
- 失败案例与 reward hacking 分析
