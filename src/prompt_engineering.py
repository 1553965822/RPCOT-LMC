from typing import Dict, Any, List, Optional
from src.utils import logger, ConfigManager
from src.retrieval_module import RiskKnowledgeRetriever

class PromptEngineer:
    """
    Module for generating prompts for the LLM
    """
    
    def __init__(self):
        """Initialize the prompt engineer with configuration"""
        self.config = ConfigManager()
        self.knowledge_retriever = RiskKnowledgeRetriever()
        self.max_tokens = int(self.config.experiment_config.get('prompt_max_tokens', 150))
    
    def generate_risk_analysis_prompt(self, paragraph: str, risk_point: str, template: str = "standard") -> str:
        """
        Generate a prompt for risk analysis
        
        Args:
            paragraph: Contract paragraph text
            risk_point: Risk point to identify
            template: Prompt template to use
            
        Returns:
            Formatted prompt
        """
        # Get risk knowledge context
        knowledge_context = self.knowledge_retriever.generate_knowledge_context(risk_point)
        
        # Select template
        if template == "standard":
            return self._generate_standard_prompt(paragraph, risk_point, knowledge_context)
        elif template == "detailed":
            return self._generate_detailed_prompt(paragraph, risk_point, knowledge_context)
        elif template == "concise":
            return self._generate_concise_prompt(paragraph, risk_point, knowledge_context)
        else:
            logger.warning(f"Unknown prompt template: {template}, using standard")
            return self._generate_standard_prompt(paragraph, risk_point, knowledge_context)
    
    def _generate_standard_prompt(self, paragraph: str, risk_point: str, knowledge_context: str) -> str:
        """
        Generate a standard prompt for risk analysis
        
        Args:
            paragraph: Contract paragraph text
            risk_point: Risk point to identify
            knowledge_context: Context information about the risk point
            
        Returns:
            Formatted prompt
        """
        prompt = f"""你是一位专业的合同风险分析专家，请分析以下合同条款是否包含"{risk_point}"风险。

风险知识背景：
{knowledge_context}

合同条款：
{paragraph}

请以JSON格式回复分析结果：
```json
{{
  "decision": "包含 或 不包含",
  "confidence": 0到1之间的浮点数，表示你的确信程度,
  "explanation": "详细解释分析过程和法律依据",
  "risk_factors": ["具体风险因素1", "具体风险因素2", ...],
  "suggestions": ["改进建议1", "改进建议2", ...]
}}
```

请确保回复格式严格遵循上述JSON结构。
"""
        return prompt
    
    def _generate_detailed_prompt(self, paragraph: str, risk_point: str, knowledge_context: str) -> str:
        """
        Generate a detailed prompt for risk analysis
        
        Args:
            paragraph: Contract paragraph text
            risk_point: Risk point to identify
            knowledge_context: Context information about the risk point
            
        Returns:
            Formatted prompt
        """
        prompt = f"""作为一位经验丰富的合同法专家和风险分析师，请对以下合同条款进行全面深入的风险分析，特别关注"{risk_point}"风险。

风险知识背景：
{knowledge_context}

合同条款：
{paragraph}

请按以下步骤进行分析：

1. 条款解读：分析条款的法律含义和意图
2. 风险识别：明确指出条款中与"{risk_point}"相关的潜在风险点
3. 法律依据：引用相关法律法规或判例支持你的分析
4. 风险评估：评估风险的严重程度和可能影响
5. 修改建议：如存在风险，提出具体、可操作的修改建议

请以JSON格式提供分析结果：
```json
{{
  "decision": "包含 或 不包含",
  "confidence": 0到1之间的浮点数,
  "explanation": "详细的法律分析和推理过程",
  "legal_basis": ["相关法律法规1", "相关法律法规2", ...],
  "risk_factors": ["具体风险因素1", "具体风险因素2", ...],
  "risk_severity": "高/中/低",
  "suggestions": ["具体修改建议1", "具体修改建议2", ...]
}}
```

请确保你的分析专业、全面、有理有据。
"""
        return prompt
    
    def _generate_concise_prompt(self, paragraph: str, risk_point: str, knowledge_context: str) -> str:
        """
        Generate a concise prompt for risk analysis
        
        Args:
            paragraph: Contract paragraph text
            risk_point: Risk point to identify
            knowledge_context: Context information about the risk point
            
        Returns:
            Formatted prompt
        """
        # Split the knowledge context into lines and get the second line for brevity
        knowledge_lines = knowledge_context.split('\n')
        brief_context = knowledge_lines[1] if len(knowledge_lines) > 1 else knowledge_context
        
        prompt = f"""请简洁分析该合同条款是否包含"{risk_point}"风险：

条款：{paragraph}

风险背景：{brief_context}

回复格式：
```json
{{
  "decision": "包含/不包含",
  "confidence": 0-1数值,
  "explanation": "简要分析",
  "suggestions": ["建议"]
}}
```"""
        return prompt
    
    def generate_batch_prompts(self, paragraphs: List[str], risk_point: str, template: str = "standard") -> List[str]:
        """
        Generate prompts for a batch of paragraphs
        
        Args:
            paragraphs: List of contract paragraph texts
            risk_point: Risk point to identify
            template: Prompt template to use
            
        Returns:
            List of formatted prompts
        """
        prompts = []
        
        for paragraph in paragraphs:
            prompt = self.generate_risk_analysis_prompt(paragraph, risk_point, template)
            prompts.append(prompt)
        
        return prompts