# Trained models (Git LFS)

Single published freeform red-team checkpoint (policies + critics). Training scratch stays in gitignored `outputs/`.

| Path | Source |
|------|--------|
| `freeform/` | `outputs/freeform-red-team/run_20260106_123424/final_model` |

## After clone

```bash
git lfs install
git lfs pull
```

## Load

```bash
python -m src.main \
  --config_path src/configs/config.json \
  --load_from_checkpoint models/freeform
```

Base HuggingFace weights from the training config are **not** included; download the configured `hf_checkpoint` separately.

Lightweight policy-only inference package: `../redteam_agent_example/`.
