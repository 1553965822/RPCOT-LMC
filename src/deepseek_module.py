from typing import Dict, Any, List, Optional
import json
import re
from src.utils import logger, ConfigManager, call_llm_api, normalize_decision

class DeepSeekAnalyzer:
    """
    Module for analyzing contract paragraphs using the DeepSeek model.
    Specialized for legal risk analysis with domain-specific prompt engineering.
    """
    
    def __init__(self):
        """Initialize the DeepSeek analyzer with configuration"""
        self.config = ConfigManager()
        self.max_tokens = int(self.config.model_config.get('deepseek_max_tokens', 350))
        self.temperature = float(self.config.model_config.get('deepseek_temperature', 0.3))
        self.model = self.config.model_config.get('deepseek_model', 'deepseek-llm')
    
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
            # Generate specialized prompt for legal analysis
            prompt = self._generate_prompt(paragraph, risk_point)
            
            # Call DeepSeek model API
            response = call_llm_api(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model=self.model
            )
            
            # Parse the response
            result = self._parse_response(response)
            
            # Add metadata
            result["model"] = "deepseek"
            result["risk_point"] = risk_point
            result["paragraph_preview"] = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            
            return result
            
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {e}")
            return {
                "decision": "",
                "confidence": 0.0,
                "explanation": f"DeepSeek分析错误: {str(e)}",
                "suggestions": [],
                "model": "deepseek"
            }
    
    def _generate_prompt(self, paragraph: str, risk_point: str) -> str:
        """
        Generate a specialized prompt for the DeepSeek model
        
        Args:
            paragraph: Contract paragraph to analyze
            risk_point: Risk point to identify
            
        Returns:
            Formatted prompt
        """
        prompt = f"""你是一位专业的合同风险分析专家，擅长识别合同中的法律风险。请分析以下合同条款是否包含"{risk_point}"风险。

合同条款：
{paragraph}

请严格按照如下JSON格式回答：
```json
{{
  "decision": "包含 或 不包含",
  "confidence": 0到1之间的浮点数,
  "reasoning": "详细解释你的分析过程和法律依据",
  "risk_factors": ["如果包含风险，列出具体风险因素，否则为空列表"],
  "suggestions": ["对合同条款的修改建议，如没有则为空列表"]
}}
```

要求：
1. 基于法律专业知识和逻辑推理
2. 给出明确的"包含"或"不包含"决策
3. 置信度分数必须是0-1之间的浮点数
4. 提供清晰的法律分析依据
"""
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the DeepSeek model
        
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
                    logger.warning("Could not find JSON in DeepSeek response")
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
            
            # Map 'reasoning' to 'explanation' for consistency
            if "reasoning" in result and "explanation" not in result:
                result["explanation"] = result["reasoning"]
            
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
            logger.warning("Failed to decode JSON from DeepSeek response")
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
            logger.error(f"Error parsing DeepSeek response: {e}")
            default_result["explanation"] = response
            return default_result