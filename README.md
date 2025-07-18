<<<<<<< HEAD
# RPCOT-LMC: Risk Point Chain-of-Thought with Language Model Collaboration

RPCOT-LMC 是一个专注于合同风险识别与分析的智能系统，结合了链式思维推理（Chain-of-Thought）和多模型协作（Language Model Collaboration）技术，实现高精度合同风险点识别与分析。

## 系统概述

该系统通过以下核心模块实现合同风险分析：

1. **数据预处理模块**：支持从多种格式（DOCX、PDF、TXT）读取并预处理合同文本
2. **知识库检索模块**：基于语义相似度检索相关风险知识
3. **链式思维推理**：使用CoT技术进行深度风险分析
4. **多模型协作**：结合DeepSeek和MiniCPM模型，通过协作方式提高分析准确性
5. **评估模块**：提供完整的评估指标计算与结果展示

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 基本用法

分析合同文件并输出结果：

```bash
python src/run_experiment.py --contract data/raw/contract1_test.docx --output results/analysis.json
```

使用不同的分析方法：

```bash
# 使用链式思维推理
python src/run_experiment.py --contract data/raw/contract1_test.docx --method cot

# 使用直接分析
python src/run_experiment.py --contract data/raw/contract1_test.docx --method direct

# 使用模型协作（默认方法）
python src/run_experiment.py --contract data/raw/contract1_test.docx --method ensemble
```

指定特定风险点进行分析：

```bash
python src/run_experiment.py --contract data/raw/contract1_test.docx --risk_points "违约责任" "管辖权"
```

## 系统架构

```
RPCOT-LMC/
│
├── config/                # 配置文件
│   └── config.yaml        # 主配置文件
│
├── data/                  # 数据目录
│   ├── raw/               # 原始合同文件
│   ├── processed/         # 预处理后的合同文本
│   ├── risk_knowledge.json # 风险知识库
│   └── ground_truth.json  # 评估用的标准答案
│
├── src/                   # 源代码
│   ├── data_preprocessing.py  # 数据预处理
│   ├── retrieval_module.py    # 知识库检索
│   ├── prompt_engineering.py  # 提示工程
│   ├── cot_reasoning.py       # 链式思维推理
│   ├── deepseek_module.py     # DeepSeek模型
│   ├── minicpm_module.py      # MiniCPM模型
│   ├── model_collaboration.py # 模型协作
│   ├── evaluation.py          # 评估模块
│   ├── utils.py               # 工具函数
│   └── run_experiment.py      # 主程序
│
└── experiments/           # 实验结果
    └── results/           # 评估结果
```

## 高级功能

### 自定义风险点

在`data/risk_knowledge.json`中定义自己的风险点和描述：

```json
{
  "risks": {
    "违约责任": {
      "description": "合同中关于违约行为及其后果的规定",
      "examples": [
        "如一方未按约定履行义务，须向对方支付合同总额10%的违约金",
        "如因甲方原因导致项目延期，每延期一日按合同总额的千分之一支付违约金"
      ]
    },
    "管辖权": {
      "description": "合同争议解决方式和管辖法院的约定",
      "examples": [
        "凡因本合同引起的或与本合同有关的任何争议，均应提交北京仲裁委员会",
        "本合同争议由合同签订地人民法院管辖"
      ]
    }
  }
}
```

### 评估合同风险分析效果

使用标准答案进行评估：

```bash
python src/run_experiment.py --contract data/raw/contract1_test.docx --ground_truth data/ground_truth.json
```

## 配置说明

主要配置文件位于`config/config.yaml`，包含以下配置项：

- **数据配置**：文件路径、知识库位置等
- **模型配置**：模型类型、参数等
- **实验配置**：分析方法、阈值等

## 输出示例

系统分析结果示例（JSON格式）：

```json
{
  "p1": {
    "违约责任": {
      "decision": "包含",
      "confidence": 0.85,
      "explanation": "该条款明确规定了...",
      "risk_factors": ["违约金过高", "责任不对等"],
      "suggestions": ["建议降低违约金比例", "明确双方责任边界"]
    }
  }
}
```

## 贡献指南

欢迎提交问题和改进建议！请通过以下步骤参与贡献：

1. Fork该仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request
=======
# RPCOT-LMC
本项目旨在研究低资源场景下的合同风险检测方法，构建一个融合检索增强、Prompt工程与Chain-of-Thought（CoT）推理的端到端风险检测框架（CoT-Risk）。
>>>>>>> eb4c9907be9743f264cb7813d342389ba2e122e8
