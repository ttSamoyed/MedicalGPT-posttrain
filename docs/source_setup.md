# MedicalGPT Source Setup

## 1. Clone official source without model weights

Run this in a normal local terminal if Codex network is unstable:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none https://github.com/shibing624/MedicalGPT.git /path/to/MedicalGPT-src
```

Fallback:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/shibing624/MedicalGPT.git /path/to/MedicalGPT-src
```

These commands clone source code only. They do not download Qwen model weights.

## 2. Sync source into this project

After clone succeeds:

```bash
cd /path/to/MedicalGPT
bash scripts/sync_medicalgpt_source.sh
```

The sync script excludes:

- `.git`
- `models`
- `outputs`
- `data`
- `runs`
- `wandb`
- `checkpoints`

It also keeps this repository's local `README.md`.

## 3. Check official entry scripts

After sync, check:

```bash
find scripts -maxdepth 2 -type f | sort
find training -maxdepth 2 -type f | sort
```

Then adapt the wrappers under `scripts/server/` to the actual official arguments.
