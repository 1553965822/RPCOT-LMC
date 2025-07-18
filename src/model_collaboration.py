from typing import Dict, Any, List, Optional
import json
from src.utils import logger, ConfigManager
from src.deepseek_module import DeepSeekAnalyzer
from src.minicpm_module import MiniCPMAnalyzer

class ModelCollaborationManager:
    """
    Module for facilitating collaboration between different models for contract risk analysis
    """
    
    def __init__(self):
        """Initialize the collaboration manager with configuration"""
        self.config = ConfigManager()
        self.deepseek_analyzer = DeepSeekAnalyzer()
        self.minicpm_analyzer = MiniCPMAnalyzer()
        self.collaboration_threshold = float(self.config.experiment_config.get('collaboration_threshold', 0.2))
    
    def analyze_paragraph(self, paragraph: str, risk_point: str) -> Dict[str, Any]:
        """
        Analyze a paragraph using both models and integrate their results
        
        Args:
            paragraph: Contract paragraph text
            risk_point: Risk point to identify
            
        Returns:
            Integrated analysis result
        """
        try:
            # Analyze with DeepSeek model
            deepseek_result = self.deepseek_analyzer.analyze_paragraph(paragraph, risk_point)
            
            # Analyze with MiniCPM model
            minicpm_result = self.minicpm_analyzer.analyze_paragraph(paragraph, risk_point)
            
            # Integrate the results
            integrated_result = self.integrate_results(deepseek_result, minicpm_result)
            
            # Add metadata
            integrated_result["paragraph"] = paragraph
            integrated_result["risk_point"] = risk_point
            integrated_result["collaboration_threshold"] = self.collaboration_threshold
            
            logger.info(f"Collaborative analysis completed for risk point: {risk_point}")
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error in collaborative analysis: {e}")
            return {
                "decision": "",
                "confidence": 0.0,
                "explanation": f"Collaborative analysis error: {str(e)}",
                "risk_factors": [],
                "suggestions": [],
                "deepseek_decision": "",
                "minicpm_decision": ""
            }
    
    def integrate_results(self, deepseek_result: Dict[str, Any], minicpm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate results from different models
        
        Args:
            deepseek_result: Analysis result from DeepSeek model
            minicpm_result: Analysis result from MiniCPM model
            
        Returns:
            Integrated analysis result
        """
        # Extract decisions and confidence scores
        minicpm_decision = minicpm_result.get("decision", "")
        deepseek_decision = deepseek_result.get("decision", "")
        
        minicpm_confidence = minicpm_result.get("confidence", 0.0)
        deepseek_confidence = deepseek_result.get("confidence", 0.0)
        
        # Calculate weighted decision based on confidence
        if minicpm_decision == deepseek_decision:
            # Both models agree
            final_decision = minicpm_decision
            final_confidence = (minicpm_confidence + deepseek_confidence) / 2
            agreement = "两个模型结果一致"
        else:
            # Models disagree, use weighted decision
            confidence_diff = abs(minicpm_confidence - deepseek_confidence)
            
            if confidence_diff > self.collaboration_threshold:
                # Use decision from model with significantly higher confidence
                if minicpm_confidence > deepseek_confidence:
                    final_decision = minicpm_decision
                    final_confidence = minicpm_confidence * 0.9  # Slightly reduce confidence due to disagreement
                    agreement = "MiniCPM置信度显著更高"
                else:
                    final_decision = deepseek_decision
                    final_confidence = deepseek_confidence * 0.9
                    agreement = "DeepSeek置信度显著更高"
            else:
                # Close confidence scores, prefer more conservative result (assume "包含" is more conservative)
                if minicpm_decision == "包含" or deepseek_decision == "包含":
                    final_decision = "包含"
                    final_confidence = max(minicpm_confidence, deepseek_confidence) * 0.8
                    agreement = "模型结果不一致，采取保守判断 (包含风险)"
                else:
                    final_decision = "不包含"
                    final_confidence = max(minicpm_confidence, deepseek_confidence) * 0.8
                    agreement = "模型结果不一致，采取保守判断 (无风险)"
        
        # Combine explanations
        minicpm_explanation = minicpm_result.get("explanation", "")
        deepseek_explanation = deepseek_result.get("explanation", "")
        
        # Combine risk factors and suggestions
        risk_factors = set()
        for factor in minicpm_result.get("risk_factors", []) + deepseek_result.get("risk_factors", []):
            if factor:  # Ignore empty strings
                risk_factors.add(factor)
        
        suggestions = set()
        for suggestion in minicpm_result.get("suggestions", []) + deepseek_result.get("suggestions", []):
            if suggestion:  # Ignore empty strings
                suggestions.add(suggestion)
        
        # Create integrated result
        integrated_result = {
            "decision": final_decision,
            "confidence": final_confidence,
            "explanation": f"模型协作分析结果：\n{agreement}\n\nMiniCPM分析：\n{minicpm_explanation}\n\nDeepSeek分析：\n{deepseek_explanation}",
            "risk_factors": list(risk_factors),
            "suggestions": list(suggestions),
            "agreement": agreement,
            "minicpm_decision": minicpm_decision,
            "deepseek_decision": deepseek_decision,
            "minicpm_confidence": minicpm_confidence,
            "deepseek_confidence": deepseek_confidence
        }
        
        return integrated_result
    
    def batch_analyze_paragraphs(self, paragraphs: List[str], risk_point: str) -> List[Dict[str, Any]]:
        """
        Analyze multiple paragraphs and return results for each
        
        Args:
            paragraphs: List of contract paragraph texts
            risk_point: Risk point to identify
            
        Returns:
            List of analysis results for each paragraph
        """
        results = []
        
        for i, paragraph in enumerate(paragraphs):
            logger.info(f"Analyzing paragraph {i+1}/{len(paragraphs)}")
            result = self.analyze_paragraph(paragraph, risk_point)
            results.append(result)
        
        return results