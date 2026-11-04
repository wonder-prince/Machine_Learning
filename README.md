# Machine Learning 实验

两个实验共用一套 Python 3.11 环境，代码、数据和评估结果分别放在 `experiments/` 下：

| 实验 | 说明 |
| --- | --- |
| [银行营销预测](experiments/bank-marketing/README.md) | 逻辑回归、随机森林和梯度提升，预测客户是否订阅定期存款 |
| [电信客户流失预测](experiments/telco-churn/README.md) | 决策树与逻辑回归对比，预测客户是否流失，并支持新客户 CSV 预测 |

## 快速开始

在仓库根目录运行：

```bash
bash setup.sh
.venv/bin/python experiments/bank-marketing/src/train.py
.venv/bin/python experiments/telco-churn/src/train.py
.venv/bin/python experiments/telco-churn/src/predict.py --input experiments/telco-churn/data/example_customers.csv
```

`setup.sh` 创建根目录的 `.venv`，安装 `requirements.txt`，并确认数据文件就位。已经安装 Conda 时，也可使用：

```bash
conda create -n ml-experiments python=3.11 -y
conda activate ml-experiments
python -m pip install -r requirements.txt
python experiments/telco-churn/src/train.py
```

各脚本默认从自身实验目录读取数据并写入 `results/`，因此可以在仓库根目录执行。`results/*.joblib` 和预测输出可重新生成，未纳入 Git；评估指标和图表已保存。VS Code 调试配置见 `.vscode/launch.json`。
