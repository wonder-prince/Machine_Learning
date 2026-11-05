# 电信客户流失预测：决策树实验

本项目依据第三章实验任务实现 IBM Telco Customer Churn 数据集上的二分类预测。PPT 的下载说明有一处写成了 Bank Marketing，但题目、字段 `InternetService`、`Contract`、`PaymentMethod` 和目标 `Churn` 均对应电信流失数据，因此这里使用 IBM 的电信数据。逻辑回归在相同数据划分上训练，用于 ROC 对比。

## 环境与运行

从仓库根目录运行。数据集已经放在 `experiments/telco-churn/data/Telco-Customer-Churn.csv`：

```bash
bash setup.sh
.venv/bin/python experiments/telco-churn/src/train.py
.venv/bin/python experiments/telco-churn/src/predict.py --input experiments/telco-churn/data/example_customers.csv
```

如使用 Conda，可按根目录 README 的说明安装相同的 `requirements.txt`，再用 `python` 运行上述两个脚本。

预测输入须含相同的 19 个特征列，`Churn` 不需要提供；`customerID` 可选。输出包含 `customerID`、流失概率 `churn_probability` 和以 0.5 为阈值的 `predicted_churn`。示例文件的三行来自原始数据，只用于演示输入格式，不代表独立验证集。不要把 `customerID` 或 `Churn` 当作预测特征。

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

### 决策树参数对照实验

运行 `.venv/bin/python experiments/telco-churn/src/analyze_tree.py` 可复现 `results/tree_parameter_analysis.json`。仍使用随机种子 42 的分层 80/20 划分，只在 5,634 个训练样本上搜索原有 64 组参数，并比较分层 3、5、10 折；测试集不参与这次参数或折数的比较。下表中除了当前参数，均只改变一项，其余固定为 `entropy / depth=5 / leaves=32 / min_leaf=10`。数字为 5 折平均验证 ROC AUC：

| 单独改变的参数 | 候选值及平均 AUC |
| --- | --- |
| `criterion` | gini: 0.82998；entropy: **0.83228** |
| `max_depth` | 3: 0.81599；**5: 0.83228**；7: 0.82590；10: 0.82551 |
| `max_leaf_nodes` | 8: 0.81896；16: 0.82867；**32: 0.83228**；64: 0.83228 |
| `min_samples_leaf` | **10: 0.83228**；30: 0.83168 |

深度为 5 时，叶节点上限改为 64 得分不变；完整训练后这棵树实际深度为 5、叶节点数为 31。32 是搜索结果中的并列优胜值，不代表独立证明 32 严格优于 64。`min_samples_leaf` 的两个候选仅差约 0.00060，不能凭这次划分认定 10 稳定优于 30。固定其他参数、把深度放宽到 7 或 10 后，平均训练 AUC 从 0.85293 上升到 0.85795/0.85802，验证 AUC 却下降到 0.82590/0.82551，符合复杂度过高可能导致过拟合的现象。

| 折数 | 完整网格所需拟合次数 | 各自最佳平均验证 AUC | 各自最佳参数（准则 / 深度 / 叶数上限 / 叶样本下限） |
| ---: | ---: | ---: | --- |
| 3 | 192 | 0.83425 | entropy / 5 / 32 / 30 |
| 5 | 320 | 0.83228 | entropy / 5 / 32 / 10 |
| 10 | 640 | 0.83026 | gini / 5 / 32 / 30 |

不同折数使用不同大小的训练折和验证折，所得平均 AUC 不宜直接当成折数的性能排名；10 折训练开销是 5 折的两倍。5 折是兼顾训练样本量（每次约 4,507 人）、验证样本量（每次约 1,127 人）和计算成本的惯常选择，不是从以上数据证明出来的唯一最佳折数。不同折数选择了不同准则和叶样本下限，提示这两项选择对评估划分敏感；需要更可靠的性能区间时可考虑重复分层交叉验证，选择流程应始终局限在训练数据中。

输出文件：

- `results/tree_parameter_analysis.json`：以上参数和折数对照的完整记录
- `results/metrics.json`：数据概况、最佳参数、交叉验证和测试指标、完整分类报告与混淆矩阵
- `results/confusion_matrix.png`：决策树混淆矩阵
- `results/confusion_matrix_logistic_regression.png`：逻辑回归混淆矩阵
- `results/roc_curves.png`：决策树与逻辑回归 ROC 对比
- `results/roc_curve_decision_tree.png`、`results/roc_curve_logistic_regression.png`：各模型单独的测试集 ROC 曲线
- `results/decision_tree.joblib`、`results/logistic_regression.joblib`：包含预处理的训练模型
- `results/predictions.csv`：运行预测命令后生成

数据来源：[IBM 官方示例仓库](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)。IBM 仓库以 Apache-2.0 许可发布；使用数据时请查看仓库的许可与说明。
