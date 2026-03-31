# Ant-Transformer 最终上线前检查（2026-03-28）

## 问题列表

1. 🔴 **测试集泄露到模型选择流程**：`run_experiment` 的参数名是 `val_loader`，但在主流程里三组模型都把 `test_loader` 传入并用于每个 epoch 的评估与“最佳模型”选择（`best_sharpe`），属于明确的测试集信息泄露（model selection on test）。
2. 🔴 **评估脚本存在直接运行时错误**：`evaluate.py` 在 `main()` 末尾使用了未定义变量 `verbose`，会触发 `NameError`，导致评估流程不稳定。
3. 🔴 **训练主入口预测导出逻辑存在结构性 bug**：`train.py` 的 `get_predictions()` 访问 `loader.dataset.df`，但 `FinancialDataset` 并未定义 `df` 字段，运行时将报错。
4. 🔴 **“Layer0 + Layer2”实验实现与声明不一致**：`create_model_b()` 实际构造的是一个全新 2 层 `TwoLayerEncoder`（连续两层），不是“4层结构中保留 Layer0 与 Layer2 的剪枝结构”，结论不可对应到声明实验假设。
5. 🟡 **“禁用剪枝”开关存在失效风险**：`train.py` 中 `train_one_epoch(..., enable_pruning)` 形参未在前向中生效，实际是否剪枝依赖 `config.enable_layer_pruning`；接口语义与真实行为不一致，易产生误用。
6. 🟡 **未固定随机种子，结果复现性不足**：训练与实验脚本均未设置 `torch / numpy / random` 随机种子与确定性选项，三模型对比容易受初始化和数据打乱随机性影响。
7. 🟡 **模型可比性存在隐藏变量**：三模型虽然训练超参表面一致，但参数规模、层数、门控分布与初始化路径不同；同时当前流程按各自“测试集最优 epoch”汇报，会进一步放大不可比性。
8. 🟢 **Sharpe 定义较简化**：`evaluate_quant` 使用月度 top20% 组合均值收益的均值/标准差，未年化、未扣无风险利率、未给出换手/成本假设；可用于内部相对比较，但不宜直接用于上线收益预期。

