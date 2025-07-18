from typing import Dict, Any, List, Optional
import re
import json
from src.utils import logger, ConfigManager, call_llm_api
from src.retrieval_module import RiskKnowledgeRetriever

class ChainOfThoughtReasoner:
    """
    Module for implementing Chain-of-Thought (CoT) reasoning for contract risk analysis
    """
    
    def __init__(self):
        """Initialize the CoT reasoner with configuration"""
        self.config = ConfigManager()
        self.knowledge_retriever = RiskKnowledgeRetriever()
        self.max_tokens = int(self.config.experiment_config.get('cot_max_tokens', 300))
    
    def construct_cot_prompt(self, paragraph: str, risk_point: str) -> str:
        """
        Construct a Chain-of-Thought prompt for contract risk analysis
        
        Args:
            paragraph: Contract paragraph to analyze
            risk_point: Risk point to identify
            
        Returns:
            Formatted CoT prompt
        """
        # Get risk knowledge context
        knowledge_context = self.knowledge_retriever.generate_knowledge_context(risk_point)
        
        # Construct CoT prompt with step-by-step reasoning
        prompt = f"""请以法律专家身份，使用逐步推理方式，分析以下合同条款是否包含"{risk_point}"风险。

风险知识背景：
{knowledge_context}

合同条款：
{paragraph}

请按照以下步骤进行分析：
1. 理解条款：简要解释条款的主要内容和法律意义
2. 识别关键要素：识别条款中与"{risk_point}"相关的关键要素
3. 法律分析：根据法律专业知识分析这些要素是否构成风险
4. 得出结论：明确判断该条款是否包含"{risk_point}"风险

最后，请以JSON格式总结你的分析结果：

```json
{{
  "decision": "包含 或 不包含",
  "confidence": 0到1之间的浮点数,
  "explanation": "详细解释分析过程和法律依据",
  "risk_factors": ["具体风险因素1", "具体风险因素2", ...],
  "suggestions": ["改进建议1", "改进建议2", ...]
}}
```
"""
        return prompt
    
    def perform_cot_analysis(self, paragraph: str, risk_point: str) -> Dict[str, Any]:
        """
        Perform Chain-of-Thought reasoning for risk analysis
        
        Args:
            paragraph: Contract paragraph to analyze
            risk_point: Risk point to identify
            
        Returns:
            Analysis result with decision, confidence, and explanation
        """
        try:
            # Construct CoT prompt
            prompt = self.construct_cot_prompt(paragraph, risk_point)
            
            # Call LLM API with CoT prompt
            response = call_llm_api(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.3,
                model="deepseek-llm"  # Use DeepSeek for CoT reasoning (typically best at reasoning)
            )
            
            # Parse the response
            result = self._parse_response(response)
            
            # Add metadata
            result["model"] = "cot_reasoning"
            result["risk_point"] = risk_point
            result["paragraph_preview"] = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            
            logger.info(f"Completed CoT reasoning for risk point: {risk_point}")
            return result
            
        except Exception as e:
            logger.error(f"Error in CoT reasoning: {e}")
            return {
                "decision": "",
                "confidence": 0.0,
                "explanation": f"CoT推理错误: {str(e)}",
                "risk_factors": [],
                "suggestions": [],
                "model": "cot_reasoning"
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the LLM
        
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
                    logger.warning("Could not find JSON in CoT response")
                    default_result["explanation"] = response
                    return default_result
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Normalize the decision
            if "decision" in result:
                decision = result["decision"].lower()
                if "包含" in decision:
                    result["decision"] = "包含"
                elif "不包含" in decision:
                    result["decision"] = "不包含"
            
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
            
            # Add the full analysis text
            result["full_analysis"] = response
            
            return result
            
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from CoT response")
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
            logger.error(f"Error parsing CoT response: {e}")
            default_result["explanation"] = response
            return default_result