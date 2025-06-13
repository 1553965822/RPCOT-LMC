# 合同风险检测

项目简介

本项目旨在研究低资源场景下的合同风险检测方法，构建一个融合检索增强、Prompt工程与Chain-of-Thought（CoT）推理的端到端风险检测框架（CoT-Risk）。该框架利用大模型进行条款审查，提高风险识别的准确性和可解释性。


## Usage
### 安装依赖
	在运行项目之前，请确保安装了必要的依赖库，可以使用以下命令进行安装：
		### pip install -r requirements.txt


## 运行实验
	在 project_root 目录下执行以下命令：
		###  python src/run_experiment.py

## 评估指标
	实验结果将自动计算以下指标：

	准确率（Accuracy）

	精确率（Precision）

	召回率（Recall）

	F1分数（F1-score）

	评估结果将输出到 experiments/results/ 目录。


## 配置文件

所有实验相关的路径、超参数配置等存放于 config/config.yaml，可根据实际情况调整。

## 贡献与反馈

如果你对本项目有任何建议或改进，请随时提交 Issue 或 Pull Request！