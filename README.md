# Ant-Transformer Financial Prediction System

[中文文档 / Chinese README](README_zh.md)

Ant-Transformer is a financial return prediction project based on a dynamic-routing Transformer design.
It introduces **cross-layer attention residuals** and a **history-driven gate** to adaptively control layer usage.

## Highlights

- **Flexible architecture**: run `layer0`, `layer0_layer2`, and full gated model variants.
- **Dynamic routing**: gate values can reduce ineffective layer computation.
- **Experiment workflow**: predefined experiment configs and batch scripts.
- **Quant evaluation**: monthly IC/RankIC and portfolio-style backtest metrics.

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run a quick check

```bash
python quick_test.py
```

### 3) Train

```bash
python train.py
```

### 4) Evaluate predictions

```bash
python evaluate.py --pred_path outputs/pred_full.csv
```

## Experiment Entrypoints

```bash
# Full model
python train.py --config configs/exp_full.yaml

# Pruned architecture
python train.py --model_arch layer0_layer2

# Single layer
python train.py --model_arch layer0

# Disable pruning behavior
python train.py --no_pruning
```

## Repository Structure

```text
.
├── train.py
├── evaluate.py
├── scripts/
├── model/
├── data/
├── configs/
├── benchmarks/
├── README.md
└── README_zh.md
```

## License

MIT. See [LICENSE](LICENSE).

---

## 👨‍💻 Team & Contact

**Project Lead:** Bowei Liu  
**Email:** [20070316lbw@gmail.com](mailto:20070316lbw@gmail.com)  
**University:** Hunan University of Information Technology (Freshman)  
**Major:** Financial Management

**Core Contributors**

- **Bowei Liu**: Architecture design, manual authorship, and result evaluation. (提供了一双手和一个脑子)
- **Gemini**: Coding MASTER, responsible for script writing, model building, and debugging. (代码编写高手)
- **Claude**: Project report auditor and conversational collaborator; raised many critical questions during research. (项目报告检查兼聊天员)
- **ChatGPT**: Project report auditor and advisor; contributed key insights to methodology. (项目报告检查)
- **GLM**: Project report auditor. Will have more important job in future.

(Names listed in no particular order; all are core forces of the project.)

---

## Disclaimer / 免责声明

The code and data in this project are for educational and research purposes only and do not constitute any investment advice. Please use with caution. 本项目的代码和数据仅供学习和研究使用，不构成任何投资建议，请谨慎使用。
