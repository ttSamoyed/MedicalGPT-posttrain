# 双 4090 服务器工作流

## 核心原则

本地负责开发和整理实验记录，借来的服务器只负责下载模型、跑训练和保存产物。

因为服务器是别的团队的机器，仓库里不要写个人姓名、本机绝对路径、私有 token 或平台专属目录。所有路径通过环境变量传入。

## 本地阶段

在本机完成：

1. 修改脚本和配置
2. 编写数据审计工具
3. 准备训练启动脚本
4. 准备评测样例
5. 写实验日志和 README

推荐本地目录写成通用形式：

```text
/path/to/MedicalGPT
```

## 远程服务器阶段

推荐远程目录：

```text
/data/medical/MedicalGPT
```

远程服务器上只做：

1. `git pull`
2. 下载模型和数据
3. 运行训练脚本
4. 保存 adapter、logs、metrics、sample outputs
5. 把必要结果拉回本地或推到私有仓库

## 连接方式

1. 使用对方提供的 SSH 登录信息连接服务器。
2. 进入项目目录，例如 `/data/medical/MedicalGPT`。
3. 不在服务器上长时间写文档，远程只做小 bug 修复和运行。
4. 离开服务器前清理 shell history 中可能出现的 token。

## 双 4090 默认策略

### DPO

- 默认：2 x RTX 4090 24GB
- 方法：QLoRA 4bit + bf16 + gradient accumulation
- 目标：先跑通 baseline，再跑清洗数据版本

### Reward Model

- 默认：2 x RTX 4090 24GB
- 目标：训练 pairwise reward model，记录 chosen/rejected accuracy

### PPO

- 默认：2 x RTX 4090 24GB
- 目标：smoke test，不追求长时间训练
- 原因：PPO 同时涉及 actor、reference、reward、critic，显存压力更大

### GRPO

- 默认：2 x RTX 4090 24GB
- 目标：小规模调通奖励函数和输出格式，再逐步增加 group size、response length 或样本量

## 每次开机前检查清单

本地先确认：

- 训练脚本已经写好
- 数据路径已经参数化
- 输出目录不会覆盖旧结果
- `reports/experiment_log.md` 已创建对应实验条目
- 预计运行时长和显存需求明确

服务器上执行：

```bash
cd /data/medical/MedicalGPT
bash scripts/server/00_env_check.sh
bash scripts/server/01_prepare_assets.sh
```
