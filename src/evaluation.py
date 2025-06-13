# src/evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging


def evaluate_metrics(y_true, y_pred, positive_label="包含"):
    """
    计算分类任务的评估指标：
    - Accuracy: 准确率
    - Precision: 精度
    - Recall: 召回率
    - F1: F1分数

    参数：
    - y_true: 真实标签列表
    - y_pred: 模型预测标签列表
    - positive_label: 定义的正类标签（默认“包含”）

    返回：
    - metrics: 包含各指标的字典
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=positive_label, average='binary')
        recall = recall_score(y_true, y_pred, pos_label=positive_label, average='binary')
        f1 = f1_score(y_true, y_pred, pos_label=positive_label, average='binary')
    except Exception as e:
        logging.error(f"计算评估指标时发生错误: {e}")
        return {}

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return metrics


if __name__ == "__main__":
    # 测试示例
    # TODO: 替换下列示例数据为实际实验中收集的真实标签和预测结果
    y_true = ["包含", "不包含", "包含", "包含", "不包含"]
    y_pred = ["包含", "包含", "包含", "不包含", "不包含"]

    metrics = evaluate_metrics(y_true, y_pred)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
