# Ant-Transformer 金融预测系统

[English README](README.md)

Ant-Transformer 是一个用于金融收益预测的研究项目，核心是动态路由 Transformer：
通过**跨层注意力残差**与**历史驱动门控（gate）**机制，自适应地决定层间信息流。

## 项目亮点

- **架构可配置**：支持 `layer0`、`layer0_layer2`、full gated 等实验结构。
- **动态路由机制**：通过 gate 控制层使用强度，减少无效计算。
- **实验流程完整**：提供预设配置与批量实验脚本。
- **量化评估闭环**：支持按月 IC/RankIC 与组合化回测指标。

## 快速开始

### 1）安装依赖

```bash
pip install -r requirements.txt
```

### 2）快速自检

```bash
python quick_test.py
```

### 3）训练模型

```bash
python train.py
```

### 4）评估结果

```bash
python evaluate.py --pred_path outputs/pred_full.csv
```

## 常用实验命令

```bash
# 完整模型
python train.py --config configs/exp_full.yaml

# 剪枝结构
python train.py --model_arch layer0_layer2

# 单层模型
python train.py --model_arch layer0

# 禁用剪枝
python train.py --no_pruning
```

## 仓库结构

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

## 许可证

MIT，详见 [LICENSE](LICENSE)。

---

## 👨‍💻 Team & Contact

**Project Lead：** Bowei Liu  
**Email：** [20070316lbw@gmail.com](mailto:20070316lbw@gmail.com)  
**University：** Hunan University of Information Technology（大一）  
**Major：** Financial Management（财务管理）

**Core Contributors**

- **Bowei Liu**：Architecture design, manual manual authorship, and result evaluation. (提供了一双手和一个脑子)
- **Gemini**：Coding MASTER, responsible for script writing, model building, and debugging. (代码编写高手)
- **Claude**：Project report auditor and conversational collaborator; raised many critical questions during research. (项目报告检查兼聊天员)
- **ChatGPT**：Project report auditor and advisor; contributed key insights to methodology. (项目报告检查)
- **GLM**：Project report auditorr. Will have more important job in future.

(Names listed in no particular order; all are core forces of the project.)

---

## Disclaimer / 免责声明

The code and data in this project are for educational and research purposes only and do not constitute any investment advice. Please use with caution. 本项目的代码和数据仅供学习和研究使用，不构成任何投资建议，请谨慎使用。
