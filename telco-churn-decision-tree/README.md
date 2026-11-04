# 电信客户流失预测：决策树实验

本项目依据第三章实验任务实现 IBM Telco Customer Churn 数据集上的二分类预测。PPT 的下载说明有一处写成了 Bank Marketing，但题目、字段 `InternetService`、`Contract`、`PaymentMethod` 和目标 `Churn` 均对应电信流失数据，因此这里使用 IBM 的电信数据。逻辑回归在相同数据划分上训练，用于 ROC 对比。

## 环境与运行

数据集已经放在 `data/Telco-Customer-Churn.csv`。在本目录运行：

```bash
conda create -n telco-churn python=3.11 -y
conda activate telco-churn
python -m pip install -r requirements.txt
python src/train.py
```

如果不使用 Conda，也可以创建 Python 3.11 虚拟环境后安装 `requirements.txt`。

训练完成后，对具有相同 19 个特征列的新客户 CSV 预测：

```bash
python src/predict.py --input data/example_customers.csv --output results/predictions.csv
```

输入列名与数据集相同，`Churn` 不需要提供；`customerID` 可选。输出包含 `customerID`、流失概率 `churn_probability` 和以 0.5 为阈值的 `predicted_churn`。示例文件的三行来自原始数据，只用于演示输入格式，不代表独立验证集。不要把 `customerID` 或 `Churn` 当作预测特征。

## 设计

- 数据按 `Churn` 分层，固定随机种子 42，80% 训练、20% 测试。测试集只用于最终评估。
- 将空白 `TotalCharges` 转为缺失值；数值特征用训练折中位数填补并标准化。无序分类字段做独热编码；`Contract` 按月付、一年、两年编码为 0、1、2。类别字段中的 `unknown` 会保留为独立类别。
- 预处理、编码和模型一起放入 `Pipeline`，在每个交叉验证训练折内重新拟合，避免测试数据或验证折的信息泄漏。
- 正类流失客户占 26.54%；决策树和逻辑回归均设置 `class_weight='balanced'`。决策树使用分层 5 折交叉验证及网格搜索，以 ROC AUC 选择 `criterion`、`max_depth`、`max_leaf_nodes` 和 `min_samples_leaf`。逻辑回归同样使用分层 5 折搜索 `C`。
- 输出 `classification_report`、混淆矩阵、ROC 曲线及 AUC。正类为 `Yes`。

## 本次完整运行结果

训练集 5,634 人，测试集 1,409 人，其中流失客户 374 人。以下 Precision、Recall、F1 均指流失类：

| 模型 | 测试 AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 决策树 | 0.8374 | 0.5262 | 0.7513 | 0.6189 |
| 逻辑回归 | 0.8406 | 0.5043 | 0.7834 | 0.6136 |

决策树最佳参数为 `criterion=entropy`、`max_depth=5`、`max_leaf_nodes=32`、`min_samples_leaf=10`。其混淆矩阵按真实类别行、预测类别列排列为 `[[782, 253], [93, 281]]`，即误报 253 人、漏报 93 人。逻辑回归的 AUC 略高，决策树在默认阈值下的流失类 Precision 和 F1 略高。AUC 是本次固定划分上的实测值，不保证在其他样本上保持不变。

输出文件：

- `results/metrics.json`：数据概况、最佳参数、交叉验证和测试指标、完整分类报告与混淆矩阵
- `results/confusion_matrix.png`：决策树混淆矩阵
- `results/roc_curves.png`：决策树与逻辑回归 ROC 对比
- `results/decision_tree.joblib`、`results/logistic_regression.joblib`：包含预处理的训练模型
- `results/predictions.csv`：运行预测命令后生成

数据来源：[IBM 官方示例仓库](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)。IBM 仓库以 Apache-2.0 许可发布；使用数据时请查看仓库的许可与说明。
