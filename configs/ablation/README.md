# IC Ablation Configs: Grouped Frequency Attention

## Goal
Compare IC/RankIC performance between baseline MHA and grouped-frequency attention.

## Config set
- `ic_baseline.yaml`: Standard MHA (no grouped attention)
- `ic_grouped_g2.yaml`: Grouped attention with 2 groups
- `ic_grouped_g4.yaml`: Grouped attention with 4 groups

## Run
```bash
python train.py --config configs/ablation/ic_baseline.yaml
python train.py --config configs/ablation/ic_grouped_g2.yaml
python train.py --config configs/ablation/ic_grouped_g4.yaml
```

## Suggested comparison table
| Config | Mean IC | Mean RankIC | Sharpe | Notes |
|---|---:|---:|---:|---|
| baseline |  |  |  | standard MHA |
| grouped_g2 |  |  |  | coarse frequency groups |
| grouped_g4 |  |  |  | finer frequency groups |
