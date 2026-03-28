@echo off
REM DynaRouter 批量实验脚本 (Windows版本)
REM 用于一次性运行所有配置的实验

echo ========================================
echo   DynaRouter 批量实验脚本
echo ========================================
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python 未安装或不在 PATH 中
    pause
    exit /b 1
)

REM 实验列表
set experiments=exp_full exp_layer0_layer2 exp_layer0 exp_no_pruning exp_tuned_gate

REM 运行所有实验
for %%e in (%experiments%) do (
    echo ========================================
    echo 运行实验: %%e
    echo ========================================
    
    python train.py --config "configs\%%e.yaml"
    
    if errorlevel 1 (
        echo 错误: 实验 %%e 失败
        pause
        exit /b 1
    )
    
    echo.
)

REM 评估所有结果
echo ========================================
echo 评估所有实验结果
echo ========================================
echo.

if exist "outputs\pred_*.csv" (
    for %%f in (outputs\pred_*.csv) do (
        echo 评估: %%f
        python evaluate.py --pred_path "%%f"
        echo.
    )
) else (
    echo 警告: 未找到预测结果文件
)

echo ========================================
echo 所有实验完成！
echo ========================================

pause
