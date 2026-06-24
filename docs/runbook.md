# MedicalGPT Post-Training Runbook

## Phase 0: Local Preparation

本地已经完成：

- 官方 MedicalGPT 源码同步
- 通用服务器 wrapper 脚本
- 数据审计脚本
- PPO/GRPO smoke 数据构造脚本
- 实验记录模板

本地可运行：

```bash
python experiments/medical_posttrain/data_audit.py \
  --input data/reward/dpo_zh_500.jsonl \
  --output reports/data_audit_dpo_zh_500.md

python experiments/medical_posttrain/build_posttrain_data.py \
  --input data/sft/medical_sft_1K_format.jsonl \
  --output data/grpo/medical_sft_grpo_smoke.jsonl \
  --task grpo \
  --limit 500
```

## Phase 1: Push Code

建议把当前仓库推到自己的 GitHub 私仓，然后在双 4090 服务器上 clone/pull。

```bash
git status
git add .
git commit -m "Add MedicalGPT post-training reproduction workflow"
```

## Phase 2: Dual RTX 4090 Environment

推荐服务器配置：

- 2 x RTX 4090 24GB
- PyTorch + CUDA 12.x
- 项目目录使用通用路径，例如 `/data/medical/MedicalGPT`
- 不在仓库中保存个人路径、姓名、token 或平台专属配置

进入远程：

```bash
cd /data/medical/MedicalGPT
bash scripts/server/00_env_check.sh
bash scripts/server/01_prepare_assets.sh
bash scripts/server/03_build_smoke_data.sh
```

`01_prepare_assets.sh` 默认只安装环境并下载 base model，不下载完整 `shibing624/medical` 数据集。确认环境和 demo 数据跑通后，再执行：

```bash
DOWNLOAD_MEDICAL_DATASET=1 bash scripts/server/01_prepare_assets.sh
```

## Phase 3: DPO Baseline

先用官方 demo 偏好数据跑通流程：

DPO baseline 使用 4bit QLoRA，默认单进程通过 `device_map=auto` 在两张 4090 上切分模型，比直接 `torchrun` 更稳。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BASE_MODEL_PATH=/data/medical/MedicalGPT/models/base/Qwen2.5-7B-Instruct \
TRAIN_FILE_DIR=/data/medical/MedicalGPT/data/reward \
MAX_STEPS=100 \
bash scripts/server/10_run_dpo_baseline.sh
```

产物：

- `outputs/dpo/qwen25-7b-baseline`
- `train_results.json`
- `eval_results.json`
- tensorboard logs

## Phase 4: Reward Model

Reward Model wrapper 也保持单进程运行。原始项目说明 RM 暂不支持 `torchrun` 多卡训练，双 4090 主要通过可见双卡和 `device_map=auto` 缓解显存压力。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BASE_MODEL_PATH=/data/medical/MedicalGPT/models/base/Qwen2.5-7B-Instruct \
TRAIN_FILE_DIR=/data/medical/MedicalGPT/data/reward \
bash scripts/server/20_run_reward_model.sh
```

产物：

- `outputs/rm/qwen25-7b-rm`
- pairwise reward loss / eval metrics

## Phase 5: PPO Smoke Test

PPO 显存更重，双 4090 先只做 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BASE_MODEL_PATH=/data/medical/MedicalGPT/models/base/Qwen2.5-7B-Instruct \
REWARD_MODEL_PATH=/data/medical/MedicalGPT/outputs/rm/qwen25-7b-rm \
MAX_STEPS=100 \
bash scripts/server/30_run_ppo_smoke.sh
```

目标只是跑通，不追求最终效果。

## Phase 6: GRPO Smoke Test

GRPO wrapper 使用 `torchrun --nproc_per_node 2`，如果只想临时用一张卡，可以设置 `CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1`。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BASE_MODEL_PATH=/data/medical/MedicalGPT/models/base/Qwen2.5-7B-Instruct \
TRAIN_FILE_DIR=/data/medical/MedicalGPT/data/grpo \
TRAIN_SAMPLES=100 \
MAX_STEPS=50 \
bash scripts/server/40_run_grpo_smoke.sh
```

先跑通官方格式奖励 + answer 奖励。后续再把 reward 改成医疗语义相似度、格式规范和 judge 评分。

## Phase 7: Evidence Collection

每次实验结束后填写：

```text
reports/experiment_log.md
```

必须保留：

- 训练命令
- GPU 型号和数量
- max steps / samples
- loss / eval metrics
- 显存占用截图
- 3-5 个推理案例
- 失败案例
