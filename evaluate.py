"""
evaluate.py — 加载已训练模型，评估 + 分析门控行为

用法：
    python evaluate.py                        # 评估 val 集
    python evaluate.py --analyze_gates        # 额外输出各层平均门控值
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from config import AntConfig
from model.ant import AntTransformer
from data.dataset import get_dataloaders


@torch.no_grad()
def evaluate_with_gates(model, loader, criterion, device, collect_gates=False):
    model.eval()
    total_loss = total_correct = total_samples = 0
    # gate_stats[layer_idx] = list of mean gate values per batch
    gate_stats = [[] for _ in range(model.config.num_layers)]

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits, _, all_gates = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += bs

        if collect_gates:
            for i, g in enumerate(all_gates):
                # g: [B, T, D] → 取 [CLS] token 的均值
                gate_stats[i].append(g[:, 0, :].mean().item())

    metrics = {
        "loss": total_loss / total_samples,
        "acc":  total_correct / total_samples,
    }
    if collect_gates:
        metrics["gate_means"] = [
            sum(vals) / len(vals) for vals in gate_stats
        ]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default=None)
    parser.add_argument("--analyze_gates", action="store_true")
    args = parser.parse_args()

    config = AntConfig()
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(config)

    model = AntTransformer(config).to(device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {config.checkpoint_path}")

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_with_gates(
        model, val_loader, criterion, device,
        collect_gates=args.analyze_gates,
    )

    print(f"Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['acc']:.4f}")

    if args.analyze_gates:
        print("\nLayer-wise mean gate values (CLS token):")
        print("  (0 = 完全跳过当前层  |  1 = 完全激活当前层)")
        for i, g_mean in enumerate(metrics["gate_means"]):
            bar = "█" * int(g_mean * 20)
            print(f"  Layer {i:02d}: {g_mean:.4f}  {bar}")


if __name__ == "__main__":
    main()
