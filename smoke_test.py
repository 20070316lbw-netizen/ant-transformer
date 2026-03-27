"""
smoke_test.py — 小蚂蚁收敛验证脚本

目标：不加载真实数据，用合成数据，跑 N steps，
      观察 loss 是否稳定下降。CPU 上几十秒内跑完。

用法：
    python smoke_test.py
    python smoke_test.py --steps 100 --d_model 64
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys

# ── 直接 import 项目里的模型，不改动任何模型代码 ──────────────────────────
from model.ant import AntTransformer


def make_dummy_batch(batch_size, seq_len, vocab_size, num_classes, device):
    """生成一批随机 token id 和随机标签"""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def run_smoke_test(args):
    device = torch.device("cpu")           # 强制 CPU，快速验证
    torch.manual_seed(42)

    # ── 极小模型配置，减少参数量，CPU 跑得动 ──────────────────────────────
    VOCAB_SIZE  = 1000    # 合成数据不需要真实词表
    NUM_CLASSES = 2
    D_MODEL     = args.d_model
    N_HEADS     = args.n_heads
    N_LAYERS    = args.n_layers
    D_FF        = args.d_ff
    MAX_SEQ_LEN = args.seq_len
    BATCH_SIZE  = args.batch_size
    STEPS       = args.steps

    from config import AntConfig

    config = AntConfig(
        vocab_size   = VOCAB_SIZE,
        d_model      = D_MODEL,
        num_heads    = N_HEADS,
        num_layers   = N_LAYERS,
        d_ff         = D_FF,
        max_seq_len  = MAX_SEQ_LEN,
        num_classes  = NUM_CLASSES,
        dropout      = 0.0,
    )

    model = AntTransformer(config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"配置: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, "
          f"d_ff={D_FF}, seq_len={MAX_SEQ_LEN}, batch={BATCH_SIZE}")
    print(f"计划跑 {STEPS} steps（合成数据，CPU）\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    model.train()

    print(f"{'Step':>6}  {'Loss':>8}  {'趋势':}")
    print("-" * 35)

    # ── 固定数据，测试过拟合能力 ──────────────────────────────────────────────
    fixed_x, fixed_y = make_dummy_batch(BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE, NUM_CLASSES, device)

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        logits, _, _ = model(fixed_x)
        loss   = criterion(logits, fixed_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 每 10 步打印一次
        if step % 10 == 0 or step == 1:
            recent = losses[-10:]
            if len(recent) >= 2:
                trend = "↓" if recent[-1] < recent[0] else ("→" if abs(recent[-1] - recent[0]) < 0.01 else "↑")
            else:
                trend = " "
            print(f"{step:>6}  {loss.item():>8.4f}  {trend}")

    # ── 判断是否收敛 ────────────────────────────────────────────────────────
    print("\n" + "=" * 35)
    first_avg = sum(losses[:10]) / 10
    last_avg  = sum(losses[-10:]) / 10
    drop      = first_avg - last_avg
    drop_pct  = drop / first_avg * 100 if first_avg > 0 else 0

    print(f"前 10 步平均 loss : {first_avg:.4f}")
    print(f"后 10 步平均 loss : {last_avg:.4f}")
    print(f"下降幅度          : {drop:.4f}  ({drop_pct:.1f}%)")

    if drop_pct > 5:
        print("\n✅ Loss 正在下降，模型可以正常收敛")
        return True
    elif drop_pct > 0:
        print("\n⚠️  Loss 略有下降，但幅度较小——可以适当增加 steps 再观察")
        return True
    else:
        print("\n❌ Loss 没有下降，架构或优化器可能存在问题，建议排查")
        return False


def parse_args():
    p = argparse.ArgumentParser(description="小蚂蚁 smoke test")
    p.add_argument("--steps",    type=int,   default=50,  help="训练步数（默认 50，够看趋势）")
    p.add_argument("--d_model",  type=int,   default=64,  help="embedding 维度（默认 64，CPU 友好）")
    p.add_argument("--n_heads",  type=int,   default=4,   help="注意力头数")
    p.add_argument("--n_layers", type=int,   default=2,   help="encoder 层数")
    p.add_argument("--d_ff",     type=int,   default=128, help="前馈层宽度")
    p.add_argument("--seq_len",  type=int,   default=16,  help="序列长度（默认 16，够小）")
    p.add_argument("--batch_size", type=int, default=8,   help="batch 大小")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run_smoke_test(args)
    sys.exit(0 if ok else 1)
