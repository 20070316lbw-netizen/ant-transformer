import matplotlib.pyplot as plt
import numpy as np
import os

# 设置风格
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. 性能对比图 (Performance Comparison)
models = ['LightGBM (Full)', 'Ant (Subset)']
mse_values = [0.020555, 0.030892]
corr_values = [0.0827, 0.0535] # LightGBM: Rank IC, Ant: Val Corr

color_mse = '#e24a33'
color_corr = '#348abd'

x = np.arange(len(models))
width = 0.35

ax1_twin = ax1.twinx()
p1 = ax1.bar(x - width/2, mse_values, width, label='MSE (Lower is Better)', color=color_mse, alpha=0.8)
p2 = ax1_twin.bar(x + width/2, corr_values, width, label='IC/Corr (Higher is Better)', color=color_corr, alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel('MSE Loss', color=color_mse)
ax1_twin.set_ylabel('IC / Correlation', color=color_corr)
ax1.set_title('Performance Benchmark Comparison')

# 合并图例
lines = [p1, p2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center')

# 2. 门控分化图 (Gate Divergence - Ant Subset)
layers = ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3']
initial_gates = [0.5, 0.5, 0.5, 0.5]
final_gates = [0.4975, 0.0788, 0.0567, 0.1123]

x_gate = np.arange(len(layers))
ax2.plot(layers, initial_gates, 'o--', color='gray', label='Initial (Standard Transformer)', alpha=0.6)
ax2.plot(layers, final_gates, 'D-', color='#8e44ad', label='After 5 Epochs (Ant)', markersize=10, linewidth=3)

for i, v in enumerate(final_gates):
    ax2.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', color='#8e44ad')

ax2.set_ylim(-0.05, 0.6)
ax2.set_ylabel('Gate Value (G)')
ax2.set_title('Ant-Transformer: Automatic Adaptive Gating')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
save_path = 'c:/DynaRouter/experiment_comparison.png'
plt.savefig(save_path, dpi=150)
print(f"Plot saved to {save_path}")
