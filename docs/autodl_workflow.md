# AutoDL + VSCode + Codex 工作流

## 核心原则

本地负责开发，AutoDL 只负责运行实验。

这样可以避免长时间租用 GPU 做代码编辑、数据检查和文档整理。

## 本地阶段

在本机完成：

1. 修改脚本和配置
2. 编写数据审计工具
3. 准备训练启动脚本
4. 准备评测样例
5. 写实验日志和 README

推荐本地目录：

```text
/Users/kristianzeng/Documents/MedicalGPT
```

## AutoDL 阶段

推荐远程目录：

```text
/root/autodl-tmp/medical/MedicalGPT
```

AutoDL 上只做：

1. `git pull`
2. 下载模型和数据
3. 运行训练脚本
4. 保存 adapter、logs、metrics、sample outputs
5. 把结果拉回本地

## VSCode Remote SSH

1. AutoDL 控制台复制 SSH 登录信息。
2. VSCode 安装 Remote SSH。
3. 连接远程机器。
4. 打开 `/root/autodl-tmp/medical/MedicalGPT`。

不要在远程长时间写文档。远程只做小 bug 修复和运行。

## 推荐 GPU 策略

### DPO

- 首选：1 x RTX 4090 24GB
- 方法：QLoRA 4bit + gradient accumulation
- 目标：先跑通 baseline，再跑清洗数据版本

### Reward Model

- 首选：1 x RTX 4090 24GB
- 目标：训练 pairwise reward model，记录 chosen/rejected accuracy

### PPO

- 首选：2 x RTX 4090 或 1 x 48GB/PRO 6000
- 目标：smoke test，不追求长时间训练
- 原因：PPO 同时涉及 actor、reference、reward、critic，显存压力更大

### GRPO

- 首选：1 x RTX 4090 小规模调通
- 如需更大 group size 或更长 response，再上 2 卡

## 每次开机前检查清单

本地先确认：

- 训练脚本已经写好
- 数据路径已经参数化
- 输出目录不会覆盖旧结果
- `reports/experiment_log.md` 已创建对应实验条目
- 预计运行时长和 GPU 预算明确

AutoDL 开机后执行：

```bash
cd /root/autodl-tmp/medical/MedicalGPT
bash scripts/autodl/00_env_check.sh
bash scripts/autodl/01_prepare_assets.sh
```

