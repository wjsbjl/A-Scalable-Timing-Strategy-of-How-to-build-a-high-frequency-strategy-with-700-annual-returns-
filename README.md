# A Scalable Timing Strategy ( How to build a high frequency strategy with 700% annual returns )

A Scalable Timing Strategy ( How to build a high frequency strategy with 700% annual returns )


Statistical Methods in Finance (2023 Spring) - Market Microstructure

This is the source code file of the project.

---

## Requirements

Our code runs in Python 3.9 with the following packages required (in alphabetical order):

- matplotlib, version >= 3.5.3
- numpy, version >= 1.23.4
- pandas, version >= 1.5.1
- scikit-learn, version >= 1.1.3
- scipy, version >= 1.9.3

It is recommended to use conda to create the virtual environment. To do so, use the following script in `src` directory to create an environment:

```shell
conda env create -f env.yml
```

To activate the new environment:

```shell
conda activate fin-stat
```

---

## 工作流程

ProbitNAdaboost.ipynb是流程文件，思路汇总.docx是大概流程梳理，回测结果.png是很离谱的结果。剩下问题太多了...先开个坑，有空再看集中火力突突哪里了

---

## Repository Structure

- README.md: this file.
- dataset: directory of dataset.
- models: directory of trained models.
- src: source code of the project.
- result: results of the model and strategy.

---

## 改进方向
1. Lasso改成交叉熵损失；
2. 更换adaboost的random_state(已做，差别不大)；
3. 降频为一分钟，或者加滤波（但是已经赚钱了，也许不再需要降噪）；
4. 抽象类，以数据处理为主，继承后写Adaboost和Probit子类，重写run_strategy。
5. Adaboost是对数据调权重，为什么有features_importances_（按照特征被弱学习器在训练中被选择的次数）

---