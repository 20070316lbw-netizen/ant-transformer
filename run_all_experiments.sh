#!/bin/bash

# DynaRouter 批量实验脚本
# 用于一次性运行所有配置的实验

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DynaRouter 批量实验脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 实验列表
declare -a experiments=(
    "exp_full"
    "exp_layer0_layer2"
    "exp_layer0"
    "exp_no_pruning"
    "exp_tuned_gate"
)

# 运行所有实验
for exp in "${experiments[@]}"; do
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}运行实验: ${exp}${NC}"
    echo -e "${YELLOW}========================================${NC}"

    python train.py --config "configs/${exp}.yaml"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 实验 ${exp} 完成${NC}"
    else
        echo -e "${RED}✗ 实验 ${exp} 失败${NC}"
        exit 1
    fi
    echo ""
done

# 评估所有结果
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}评估所有实验结果${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for pred in outputs/pred_*.csv; do
    echo -e "${YELLOW}评估: $(basename $pred)${NC}"
    python evaluate.py --pred_path "$pred"
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
