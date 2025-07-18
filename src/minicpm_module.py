from typing import Dict, Any, List, Optional
import json
import re
from src.utils import logger, ConfigManager, call_llm_api, normalize_decision

class MiniCPMAnalyzer:
    """
    Module for analyzing contract paragraphs using the MiniCPM model.
    Provides domain-specific risk analysis with fine-tuned legal expertise.
    """
    
    def __init__(self):
        """Initialize the MiniCPM analyzer with configuration"""
        self.config = ConfigManager()
        self.max_tokens = int(self.config.model_config.get('minicpm_max_tokens', 400))
        self.temperature = float(self.config.model_config.get('minicpm_temperature', 0.2))
        self.model = self.config.model_config.get('minicpm_model', 'minicpm')
    
    def analyze_paragraph(self, paragraph: str, risk_point: str) -> Dict[str, Any]:
        """
        Analyze a contract paragraph for a specific risk point
        
        Args:
            paragraph: Contract paragraph to analyze
            risk_point: Risk point to identify
            
        Returns:
            Analysis result with decision, confidence, and explanation
        """
        try:
            # Generate specialized prompt
            prompt = self._generate_prompt(paragraph, risk_point)
            
            # Call MiniCPM model API
            response = call_llm_api(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model=self.model
            )
            
            # Parse the response
            result = self._parse_response(response)
            
            # Add metadata
            result["model"] = "minicpm"
            result["risk_point"] = risk_point
            result["paragraph_preview"] = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            
            return result
            
        except Exception as e:
            logger.error(f"Error in MiniCPM analysis: {e}")
            return {
                "decision": "",
                "confidence": 0.0,
                "explanation": f"MiniCPM分析错误: {str(e)}",
                "suggestions": [],
                "model": "minicpm"
            }
    
    def _generate_prompt(self, paragraph: str, risk_point: str) -> str:
        """
        Generate a specialized prompt for the MiniCPM model
        
        Args:
            paragraph: Contract paragraph to analyze
            risk_point: Risk point to identify
            
        Returns:
            Formatted prompt
        """
        prompt = f"""作为一位专业的合同风险分析专家，你擅长识别合同中的法律风险，特别是在"{risk_point}"方面的风险评估。请分析以下合同条款：

合同条款：
{paragraph}

请使用你的法律专业知识，判断该条款是否包含"{risk_point}"风险，并以JSON格式回复：

```json
{{
  "decision": "包含 或 不包含",
  "confidence": 0到1之间的数值，表示你的确信程度,
  "explanation": "详细解释你的分析过程，包括法律依据、风险点分析等",
  "risk_factors": ["具体风险因素1", "具体风险因素2", ...],
  "suggestions": ["改进建议1", "改进建议2", ...]
}}
```

请注意：
1. 要基于法律实践和专业知识进行分析
2. 提供明确的"包含"或"不包含"风险的判断
3. 给出0-1之间的置信度评分
4. 如果包含风险，请列出具体风险因素
5. 如有必要，提出合同条款的修改建议
"""
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the MiniCPM model
        
        Args:
            response: Raw response from the model
            
        Returns:
            Structured analysis result
        """
        default_result = {
            "decision": "",
            "confidence": 0.0,
            "explanation": "",
            "risk_factors": [],
            "suggestions": []
        }
        
        try:
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the code block markers
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    logger.warning("Could not find JSON in MiniCPM response")
                    default_result["explanation"] = response
                    return default_result
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Normalize the decision
            if "decision" in result:
                if "包含" in result["decision"]:
                    result["decision"] = "包含"
                elif "不包含" in result["decision"]:
                    result["decision"] = "不包含"
                else:
                    result["decision"] = normalize_decision(result["decision"])
            
            # Ensure confidence is a float between 0 and 1
            if "confidence" in result:
                try:
                    result["confidence"] = float(result["confidence"])
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except (ValueError, TypeError):
                    result["confidence"] = 0.5  # Default to medium confidence if parsing fails
            
            # Ensure risk_factors and suggestions are lists
            if "risk_factors" in result and not isinstance(result["risk_factors"], list):
                if isinstance(result["risk_factors"], str):
                    result["risk_factors"] = [result["risk_factors"]]
                else:
                    result["risk_factors"] = []
            
            if "suggestions" in result and not isinstance(result["suggestions"], list):
                if isinstance(result["suggestions"], str):
                    result["suggestions"] = [result["suggestions"]]
                else:
                    result["suggestions"] = []
            
            # Fill in any missing fields with defaults
            for key, value in default_result.items():
                if key not in result:
                    result[key] = value
            
            return result
            
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from MiniCPM response")
            # Try to extract decision and confidence using regex
            decision_match = re.search(r'decision["\s:]+(包含|不包含)', response, re.IGNORECASE)
            if decision_match:
                default_result["decision"] = decision_match.group(1)
            
            confidence_match = re.search(r'confidence["\s:]+(\d+\.\d+|\d+)', response, re.IGNORECASE)
            if confidence_match:
                try:
                    default_result["confidence"] = float(confidence_match.group(1))
                except ValueError:
                    pass
            
            default_result["explanation"] = response
            return default_result
            
        except Exception as e:
            logger.error(f"Error parsing MiniCPM response: {e}")
            default_result["explanation"] = response
            return default_result
    
    def integrate_with_deepseek(self, minicpm_result: Dict[str, Any], deepseek_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate MiniCPM results with DeepSeek results
        
        Args:
            minicpm_result: Analysis result from MiniCPM
            deepseek_result: Analysis result from DeepSeek
            
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
            threshold = 0.25  # Threshold for significant confidence difference
            
            if confidence_diff > threshold:
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
                # Close confidence scores, prefer more conservative result
                if minicpm_decision == "包含" or deepseek_decision == "包含":
                    final_decision = "包含"
                    final_confidence = max(minicpm_confidence, deepseek_confidence) * 0.8
                    agreement = "模型结果不一致，采取保守判断 (包含风险)"
                else:
                    final_decision = "不包含"
                    final_confidence = max(minicpm_confidence, deepseek_confidence) * 0.8
                    agreement = "模型结果不一致，采取保守判断 (无风险)"
        
        # Combine explanations, risk factors, and suggestions
        minicpm_explanation = minicpm_result.get("explanation", "")
        deepseek_explanation = deepseek_result.get("explanation", "") or deepseek_result.get("reasoning", "")
        
        risk_factors = set()
        for factor in minicpm_result.get("risk_factors", []) + deepseek_result.get("risk_factors", []):
            if factor:
                risk_factors.add(factor)
        
        suggestions = set()
        for suggestion in minicpm_result.get("suggestions", []) + deepseek_result.get("suggestions", []):
            if suggestion:
                suggestions.add(suggestion)
        
        # Construct integrated result
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