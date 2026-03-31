import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AntConfig:
    # ── 基本架构 ────────────────────────────────────────────────
    model_type: str = "text"  # "text" | "financial"
    model_arch: str = "full"  # "full" | "layer0_layer2" | "layer0"
    input_dim: int = 6  # 仅用于 financial 模式
    vocab_size: int = 30522  # bert-base-uncased 词表大小
    max_seq_len: int = 128
    d_model: int = 256  # 隐层维度
    num_heads: int = 8  # 自注意力头数
    num_layers: int = 4  # 默认层数
    d_ff: int = 1024  # FFN 内层维度
    enable_layer_pruning: bool = True  # 是否启用层裁剪功能
    use_grouped_freq_attention: bool = False  # 是否启用频率分组注意力
    num_head_groups: int = 4  # 头部分组数（仅 grouped attention）
    group_mix_coeff: float = 0.1  # 组间信息混入系数

    def update_by_arch(self):
        """根据 model_arch 自动调整架构参数"""
        arch_map = {"full": 4, "layer0_layer2": 2, "layer0": 1}
        if self.model_arch in arch_map:
            self.num_layers = arch_map[self.model_arch]
            print(
                f">>> [Config] 架构切换至 {self.model_arch}: num_layers = {self.num_layers}"
            )

    # ── 小蚂蚁专属超参 ──────────────────────────────────────────
    cross_layer_heads: int = 4  # 跨层 Attention 头数
    gate_hidden_dim: int = 64  # 历史门控 MLP 隐层

    # ── 正则化 ──────────────────────────────────────────────────
    dropout: float = 0.1

    # ── 任务 ────────────────────────────────────────────────────
    num_classes: int = 2  # SST-2: positive / negative

    # ── 训练 ────────────────────────────────────────────────────
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    gate_lambda: float = 0.01  # 门控 L1 正则权重（鼓励稀疏/跳过层）

    # ── 数据 ────────────────────────────────────────────────────
    train_end: str = "2023-12-31"
    val_end: str = "2024-12-31"
    tokenizer_name: str = "bert-base-uncased"
    checkpoint_path: str = "ant_best.pt"
    use_dummy_data: bool = False
    subset_size: int = None  # 可选：限制数据集大小以加速实验

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        """从 YAML 文件加载配置"""
        config_dict = {}
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        config = cls()

        # 加载模型配置
        if "model" in config_dict:
            for k, v in config_dict["model"].items():
                setattr(config, k, v)

        # 加载数据配置
        if "data" in config_dict:
            for k, v in config_dict["data"].items():
                setattr(config, k, v)

        # 加载训练配置
        if "training" in config_dict:
            for k, v in config_dict["training"].items():
                setattr(config, k, v)

        # 加载实验配置
        if "experiment" in config_dict:
            for k, v in config_dict["experiment"].items():
                setattr(config, k, v)

        # 加载高级配置
        if "advanced" in config_dict:
            for k, v in config_dict["advanced"].items():
                setattr(config, k, v)

        return config

    def update_by_arch(self):
        """根据 model_arch 自动调整架构参数"""
        arch_map = {"full": 4, "layer0_layer2": 2, "layer0": 1}
        if self.model_arch in arch_map:
            self.num_layers = arch_map[self.model_arch]
            print(
                f">>> [Config] 架构切换至 {self.model_arch}: num_layers = {self.num_layers}"
            )

    def validate(self):
        """执行硬性配置校验，避免静默错误"""
        # 1. Attention 分头校验
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model({self.d_model}) 必须能被 num_heads({self.num_heads}) 整除"
            )
        if self.d_model % self.cross_layer_heads != 0:
            raise ValueError(
                f"d_model({self.d_model}) 必须能被 cross_layer_heads({self.cross_layer_heads}) 整除"
            )
        if self.use_grouped_freq_attention:
            if self.num_heads % self.num_head_groups != 0:
                raise ValueError(
                    f"num_heads({self.num_heads}) 必须能被 num_head_groups({self.num_head_groups}) 整除"
                )

        # 2. 模式校验
        if self.model_type not in ["text", "financial"]:
            raise ValueError(f"不支持的 model_type: {self.model_type}")
        if self.model_arch not in ["full", "layer0_layer2", "layer0"]:
            raise ValueError(f"不支持的 model_arch: {self.model_arch}")

        # 3. 维度正数校验
        if any(
            v <= 0
            for v in [self.d_model, self.num_layers, self.num_heads, self.max_seq_len]
        ):
            raise ValueError("维度、层数、头数和序列长度必须为正数")

        print(">>> [Config] 校验通过：架构参数合法。")
