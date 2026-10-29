# Machine_Learning

## 银行营销结果预测实验

依据《机器学习 实验手册》复现 UCI Bank Marketing 数据集上的分类实验。手册明确指定逻辑回归，同时要求比较三个模型；这里补充随机森林和直方图梯度提升作为对照模型。三个模型都用网格搜索和分层 5 折交叉验证，以 ROC AUC 选择超参数。

完整实验方法、结果分析与调试记录见 [银行营销逻辑回归实验报告](银行营销逻辑回归实验报告.md)。

## 运行

本项目不需要 Docker。clone 后，在 macOS/Linux 上安装 Python 3.11，然后运行：

```bash
git clone https://github.com/wonder-prince/Machine_Learning.git
cd Machine_Learning
bash setup.sh
.venv/bin/python src/train.py
```

`setup.sh` 会创建 `.venv`、安装固定版本的依赖、检查依赖兼容性，并确认数据文件就位；安装了 `uv` 时会用它创建虚拟环境和安装依赖，重复运行也可以。仓库已包含手册指定的 `bank-additional-full.csv`，因此通常不需要再次下载。若数据文件缺失，脚本会从 UCI 下载；也可以从 [UCI 数据集页面](https://archive.ics.uci.edu/dataset/222/bank+marketing) 手动下载 `bank-additional.zip`，解压后将 CSV 放在 `data/bank-additional-full.csv`。

如需先检查安装和流程，可运行 `.venv/bin/python src/train.py --quick --cv-folds 2`。正式结果请使用不带 `--quick` 的命令。

在 VS Code 中打开本项目文件夹后，`.vscode` 配置会选用项目内的 `.venv` 解释器。安装 Microsoft Python 扩展后，可在“运行和调试”中选择“快速检查”或“完整实验”。

## 输出

- 控制台：各模型最佳超参数、交叉验证与测试集 AUC、`classification_report`
- `results/metrics.json`：机器可读指标、数据概况、无权重逻辑回归对照与混淆矩阵
- `results/confusion_matrices.png`：三个模型的混淆矩阵
- `results/roc_curves.png`：三个模型的 ROC 曲线与 AUC
- `results/*.joblib`：训练好的完整预处理与模型管道

## 本次完整运行结果

使用全部 41,188 条记录、分层 80%/20% 划分和 5 折交叉验证，测试集有 8,238 条记录，其中 928 条为正类。以下 Precision、Recall 和 F1 均针对正类 `yes`：

| 模型 | 测试集 AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 逻辑回归 | 0.9440 | 0.4493 | 0.9116 | 0.6019 |
| 随机森林 | 0.9546 | 0.5521 | 0.8341 | 0.6644 |
| 直方图梯度提升 | 0.9551 | 0.4618 | 0.9386 | 0.6190 |

三种模型都达到手册提出的 AUC ≥ 0.94 目标。随机森林在默认 0.5 阈值下正类 Precision 和 F1 较高；梯度提升的正类 Recall 较高。详细参数和混淆矩阵见 `results/metrics.json`。

随机种子固定为 42；数据按 80%/20% 分层划分。`unknown` 按手册允许的方式保留为单独类别；类别特征做独热编码，数值特征做中位数填补和标准化。预处理始终在交叉验证的训练折内拟合。逻辑回归使用 `class_weight='balanced'`；另两个模型也启用类别权重。指标均在独立测试集上计算，分类阈值为 0.5。

**解释限制：**手册指定的数据包含 `duration`（最后一次通话时长）。它只有通话结束后才知道，因此本实验评估的是通话后的预测，不适合直接用于通话前客户筛选。手册提出三个模型 AUC 均达到 0.94 的目标；实际结果以运行输出为准，代码不会人为调整测试集或阈值来达到目标。

数据来源：Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 。数据采用 CC BY 4.0 许可。
