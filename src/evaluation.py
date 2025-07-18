from typing import Dict, Any, List, Optional
import json
import os
from src.utils import logger, ConfigManager, get_absolute_path, create_directory_if_not_exists

class RiskAnalysisEvaluator:
    """
    Module for evaluating the performance metrics of risk analysis
    """
    
    def __init__(self):
        """Initialize the evaluator with configuration"""
        self.config = ConfigManager()
        self.results_dir = get_absolute_path(self.config.experiment_config.get('results_dir', 'experiments/results'))
        self.positive_label = self.config.experiment_config.get('evaluation_positive_label', '包含')
        create_directory_if_not_exists(self.results_dir)
    
    def calculate_metrics(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics for risk analysis results
        
        Args:
            predictions: List of prediction results
            ground_truth: List of ground truth labels (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "total_paragraphs": len(predictions),
            "risk_identified_count": 0,
            "no_risk_count": 0,
            "inconclusive_count": 0,
            "average_confidence": 0.0,
            "high_confidence_count": 0,
            "low_confidence_count": 0
        }
        
        total_confidence = 0.0
        high_confidence_threshold = 0.7
        low_confidence_threshold = 0.4
        
        # Calculate basic metrics
        for pred in predictions:
            decision = pred.get("decision", "")
            confidence = pred.get("confidence", 0.0)
            
            if decision == self.positive_label:
                metrics["risk_identified_count"] += 1
            elif decision and decision != self.positive_label:
                metrics["no_risk_count"] += 1
            else:
                metrics["inconclusive_count"] += 1
            
            total_confidence += confidence
            
            if confidence >= high_confidence_threshold:
                metrics["high_confidence_count"] += 1
            elif confidence <= low_confidence_threshold:
                metrics["low_confidence_count"] += 1
        
        # Calculate averages
        if len(predictions) > 0:
            metrics["average_confidence"] = total_confidence / len(predictions)
            metrics["risk_percentage"] = (metrics["risk_identified_count"] / len(predictions)) * 100
            metrics["high_confidence_percentage"] = (metrics["high_confidence_count"] / len(predictions)) * 100
            metrics["low_confidence_percentage"] = (metrics["low_confidence_count"] / len(predictions)) * 100
        
        # Calculate accuracy metrics if ground truth is available
        if ground_truth and len(ground_truth) == len(predictions):
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for pred, truth in zip(predictions, ground_truth):
                pred_decision = pred.get("decision", "")
                truth_decision = truth.get("decision", "")
                
                if pred_decision == self.positive_label and truth_decision == self.positive_label:
                    true_positives += 1
                elif pred_decision == self.positive_label and truth_decision != self.positive_label:
                    false_positives += 1
                elif pred_decision != self.positive_label and truth_decision != self.positive_label:
                    true_negatives += 1
                elif pred_decision != self.positive_label and truth_decision == self.positive_label:
                    false_negatives += 1
            
            # Calculate precision, recall, and F1 score
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / len(predictions) if len(predictions) > 0 else 0
            
            metrics["true_positives"] = true_positives
            metrics["false_positives"] = false_positives
            metrics["true_negatives"] = true_negatives
            metrics["false_negatives"] = false_negatives
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1_score
            metrics["accuracy"] = accuracy
        
        return metrics
    
    def save_evaluation_results(self, metrics: Dict[str, float], predictions: List[Dict[str, Any]], experiment_name: str) -> str:
        """
        Save evaluation results to file
        
        Args:
            metrics: Dictionary of evaluation metrics
            predictions: List of prediction results
            experiment_name: Name of the experiment
            
        Returns:
            Path to the saved results file
        """
        # Create results directory if it doesn't exist
        results_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        
        # Prepare results data
        results_data = {
            "metrics": metrics,
            "predictions": predictions
        }
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved evaluation results to {results_path}")
            return results_path
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return ""
    
    def generate_evaluation_report(self, metrics: Dict[str, float], experiment_name: str) -> str:
        """
        Generate a human-readable evaluation report
        
        Args:
            metrics: Dictionary of evaluation metrics
            experiment_name: Name of the experiment
            
        Returns:
            Formatted evaluation report
        """
        report = f"## 风险分析评估报告: {experiment_name}\n\n"
        
        report += "### 基本统计\n"
        report += f"* 分析段落总数: {metrics.get('total_paragraphs', 0)}\n"
        report += f"* 识别出风险的段落数: {metrics.get('risk_identified_count', 0)} ({metrics.get('risk_percentage', 0):.1f}%)\n"
        report += f"* 无风险的段落数: {metrics.get('no_risk_count', 0)}\n"
        report += f"* 结果不确定的段落数: {metrics.get('inconclusive_count', 0)}\n\n"
        
        report += "### 置信度分析\n"
        report += f"* 平均置信度: {metrics.get('average_confidence', 0):.2f}\n"
        report += f"* 高置信度结果数 (≥0.7): {metrics.get('high_confidence_count', 0)} ({metrics.get('high_confidence_percentage', 0):.1f}%)\n"
        report += f"* 低置信度结果数 (≤0.4): {metrics.get('low_confidence_count', 0)} ({metrics.get('low_confidence_percentage', 0):.1f}%)\n\n"
        
        # Add accuracy metrics if available
        if "accuracy" in metrics:
            report += "### 准确度评估\n"
            report += f"* 准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}\n"
            report += f"* 精确率 (Precision): {metrics.get('precision', 0):.4f}\n"
            report += f"* 召回率 (Recall): {metrics.get('recall', 0):.4f}\n"
            report += f"* F1分数: {metrics.get('f1_score', 0):.4f}\n\n"
            
            report += "### 混淆矩阵\n"
            report += "```\n"
            report += f"真正例 (TP): {metrics.get('true_positives', 0)}\n"
            report += f"假正例 (FP): {metrics.get('false_positives', 0)}\n"
            report += f"真负例 (TN): {metrics.get('true_negatives', 0)}\n"
            report += f"假负例 (FN): {metrics.get('false_negatives', 0)}\n"
            report += "```\n"
        
        return report