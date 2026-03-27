# 🐜 Ant Transformer

A custom Transformer variant with two architectural innovations:

1. **Cross-Layer Attention Residual** — each layer can attend to hidden states from *all* previous layers, not just the immediately preceding one.
2. **History-driven Gate** — a learnable gate decides how much to activate the current layer vs. skip it, based on how much useful information was found in the history.

---

## Architecture Overview

```
Input
  └─► Embedding + PositionalEncoding
        └─► AntEncoder (N × AntLayer)
              │
              ├─ Layer 0: SelfAttn → CrossLayerAttn([])     → Gate → h_0
              ├─ Layer 1: SelfAttn → CrossLayerAttn([h_0])  → Gate → h_1
              ├─ Layer 2: SelfAttn → CrossLayerAttn([h_0,h_1]) → Gate → h_2
              └─ ...
        └─► [CLS] pooling
              └─► Classifier → logits
```

### AntLayer Forward Pass

```
h_input
  │
  ├─[1]─► StandardSelfAttention         → sa_out
  │
  ├─[2]─► CrossLayerAttention(prev_hiddens) → cross_out
  │             Q = sa_out
  │             K = V = cat(prev_hiddens, dim=1)
  │
  ├─[3]─► LayerNorm(sa_out + cross_out) → combined
  │             └─► FeedForward         → ffn_out
  │
  └─[4]─► HistoryGate(cross_out, ffn_out, h_input)
               g = sigmoid(MLP(cross_out))
               h_out = g * ffn_out + (1-g) * h_input
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on SST-2
python train.py

# Train with custom hyperparameters
python train.py --epochs 5 --lr 2e-4 --d_model 128

# Evaluate + analyze gate values
python evaluate.py --analyze_gates
```

---

## Project Structure

```
ant-transformer/
├── config.py          # All hyperparameters (AntConfig dataclass)
├── train.py           # Training loop
├── evaluate.py        # Evaluation + gate analysis
├── model/
│   ├── ant.py         # AntTransformer (full model)
│   ├── encoder.py     # AntEncoder (layer stack with history)
│   ├── layer.py       # AntLayer (single layer)
│   ├── attention.py   # StandardSelfAttention + CrossLayerAttention
│   └── gate.py        # HistoryGate
└── data/
    └── dataset.py     # SST-2 dataset + DataLoader
```

---

## Default Config

| Param | Value | Note |
|---|---|---|
| `d_model` | 256 | Hidden dim |
| `num_layers` | 6 | Transformer layers |
| `num_heads` | 8 | Self-attention heads |
| `cross_layer_heads` | 4 | Cross-layer attention heads |
| `d_ff` | 1024 | FFN inner dim |
| `gate_hidden_dim` | 64 | Gate MLP hidden dim |
| `max_seq_len` | 128 | Max token length |
| `batch_size` | 32 | Training batch size |
| `lr` | 1e-4 | Peak learning rate |
| `epochs` | 10 | Training epochs |

---

## Math Reference (Work in Progress)

Each module has mathematical annotations in its docstring. The derivation order for studying:

1. `attention.py` — Scaled Dot-Product Attention, Multi-Head Attention
2. `attention.py` — Cross-Layer Attention (Q from current, K/V from history stack)
3. `gate.py` — History-driven Gate (sigmoid MLP + soft skip connection)
4. `layer.py` — Full AntLayer composition
5. `ant.py` — Positional Encoding (sinusoidal)
